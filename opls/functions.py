import copy
import logging
import os
import pickle
import re
from itertools import permutations
from openbabel import openbabel as ob
from typing import Union

import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from opls._misc import (
    OPLSAtom,
    OPLSBond,
    OPLSAngle,
    OPLSDihedral,
    OPLSImproper
)
from misc.logger import logger

__this_dir__ = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__this_dir__, 'resources', 'gmx', 'non_bond_rules.dat'), 'rb') as f:
    GMXRules = pickle.load(f)

def get_submol_with_center_first(mol, center_atom_idx, radius):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom_idx, useHs=True)
    atom_map = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
    iatom_map = {v: k for k, v in atom_map.items()}
    return submol, iatom_map  # atom_map: {orig: submol}, iatom_map: {submol: orig}


def get_submol_rad_n(mol: Union[Chem.RWMol, rdkit.Chem.rdchem.Mol],
                     radius: int, atom: Chem.Atom) -> tuple[Chem.Mol, dict, dict, str]:
    if mol.GetNumAtoms() == 1:
        return mol, None, None, Chem.MolToSmiles(mol)

    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)
    if not env:
        return
    amap = {}
    sub_mol = Chem.PathToSubmol(mol, env, atomMap=amap)
    sub_smi = Chem.MolToSmiles(sub_mol, rootedAtAtom=amap[atom.GetIdx()], canonical=False)
    return sub_mol, amap, env, sub_smi


def submol2graph(mol: Union[Chem.RWMol, rdkit.Chem.rdchem.Mol],
                 atom: Chem.Atom, radius: int) -> Union[None, tuple[nx.Graph, str]]:
    r"""The sub-env for atom, also can be used as hash with WL-hash."""
    g = nx.Graph()
    sa_prop = {}
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)
    if not env:
        return
    amap = {}
    sub_mol = Chem.PathToSubmol(mol, env, atomMap=amap)
    sub_smi = Chem.MolToSmiles(sub_mol, rootedAtAtom=amap[atom.GetIdx()], canonical=False)  # , allHsExplicit=True)
    nm = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 1:
            nm += nbr.GetAtomicNum()
    h = atom.GetHybridization()
    for a in amap:
        atom_ = mol.GetAtomWithIdx(a)
        h_ = atom_.GetHybridization()
        nm_ = 0
        for nbr in atom_.GetNeighbors():
            if nbr.GetAtomicNum() != 1:
                nm_ += nbr.GetAtomicNum()

        # sa_prop[amap[a]] = (atom.GetAtomicNum(), int(atom.GetIsAromatic()),
        #                    int(atom.IsInRing()), h.real, h.imag, atom_.GetAtomicNum(),
        #                    int(atom_.GetIsAromatic()), int(atom_.IsInRing()), nm, h_.real, h_.imag)
        sa_prop[amap[a]] = (atom.GetAtomicNum(), int(atom.GetIsAromatic()), int(atom.IsInRing()),
                            atom_.GetAtomicNum(), int(atom_.GetIsAromatic()), int(atom_.IsInRing()))

    for bond in sub_mol.GetBonds():
        ai = bond.GetBeginAtom()
        aj = bond.GetEndAtom()
        g.add_node(ai.GetIdx(), desc=sa_prop[ai.GetIdx()])
        g.add_node(aj.GetIdx(), desc=sa_prop[aj.GetIdx()])
        g.add_edge(ai.GetIdx(), aj.GetIdx(), bo=f'{bond.GetBondTypeAsDouble():.1f}')
    return g, sub_smi


def atom_hash_func(mol: Chem.Mol, radius=6, n_bits=2048) -> tuple[dict[int, np.ndarray], bytes]:
    """
    Compute a binary Morgan fingerprint of length n_bits for each atom in the molecule.
    Returns a dict: atom_idx -> numpy array of shape (n_bits,), dtype=int (0/1).
    Use this as atom db_hash value.
    """
    # Create Morgan fingerprint generator
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    # Get fingerprint with additional output that includes atom contributions
    # fp = fpgen.GetFingerprint(mol)

    # Get information about which atoms contribute to which bits
    # This returns a list of tuples: (bit_id, [(atom_id, radius), ...])
    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()
    fp_with_info = fpgen.GetFingerprint(mol, additionalOutput=additional_output)

    # Get the bit info map
    bit_info = additional_output.GetBitInfoMap()

    # Initialize per-atom fingerprints
    atom_fps = {
        atom.GetIdx(): np.zeros(n_bits, dtype=int)
        for atom in mol.GetAtoms()
    }

    # Fill in the bits for each atom according to bit_info
    for bit_id, atom_info_list in bit_info.items():
        for atom_idx, radius in atom_info_list:
            atom_fps[atom_idx][bit_id] = 1

    return atom_fps, fp_with_info.ToBinary()


def _get_stat(rdmol: Chem.Mol, atom_idx: int) -> str:
    atom = rdmol.GetAtomWithIdx(atom_idx)
    ret = atom.GetSymbol()
    stat = get_submol_rad_n(rdmol, 2, atom)
    if stat is None:
        stat = get_submol_rad_n(rdmol, 1, atom)
    ret = stat or ret
    return ret


def _build_hash(rdmol: Chem.Mol):
    logger.debug(f"Building stat...")
    # Dict: {atom_idx, np.ndarray}
    return atom_hash_func(rdmol)


def match_atom_by_boss_db(rdmol: Chem.Mol, hashes: dict, database, cache: dict, missing_atoms):
    params = {}
    missing = set()
    for atom_idx in missing_atoms:
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_stat = [0]
        if logger.level <= logging.DEBUG:
            atom_stat = _get_stat(rdmol, atom.GetIdx())
        idx = atom.GetIdx()
        atom_hash = hashes[idx]  # now np.ndarray
        hash_str = np.packbits(atom_hash).tobytes()
        if cache.get(hash_str) is not None:
            if cache.get(hash_str):
                ret_atom = cache.get(hash_str)
                params[atom.GetIdx()] = ret_atom
                logger.debug(f"Found atom {atom.GetIdx()} in CACHE. "
                             f"{atom.GetIdx()}: {atom.GetSymbol()}, *{atom_stat[-1]} as"
                             f" {ret_atom.bond_type}")
            else:
                missing.add(atom.GetIdx())
                logger.debug(f"Atom {atom.GetIdx()} in CACHE marked NOT IN database.")
            continue
        res = database.search('atom', hash_str=hash_str)
        if res:
            ret = res[0]
            ret_atom = OPLSAtom(opls_num=ret.opls_num,
                                element=atom.GetSymbol(),
                                bond_type=ret.bond_type,
                                mass=ret.mass,
                                sigma=ret.sigma,
                                epsilon=ret.epsilon,
                                charge=ret.charge,
                                ptype=ret.ptype
                                )
            params[atom.GetIdx()] = ret_atom

            logger.debug(f"Found from boss database for atom "
                         f"{atom.GetIdx()}: {atom.GetSymbol()}, *{atom_stat[-1]} as"
                         f" {ret.bond_type}")
        else:
            ret_atom = False
            missing.add(atom.GetIdx())
            logger.debug(f"{atom.GetIdx()}: {atom.GetSymbol()}, *{atom_stat[-1]} not found in database")
        cache[hash_str] = copy.deepcopy(ret_atom)
    return params, missing


def _hash_str(hashes: list[np.ndarray]) -> bytes:
    if len(hashes) == 2:
        p1 = hashes[0] ^ hashes[1]
        p2 = np.roll(hashes[1] & hashes[0], 13)
    elif len(hashes) == 3:
        p1 = hashes[0] ^ hashes[2]
        p2 = np.roll(hashes[1], 13)
    else:  # must be 4
        assert len(hashes) == 4, ValueError("!")
        p1 = (hashes[0] ^ hashes[1]) ^ (hashes[2] ^ hashes[3])
        p2 = np.roll(hashes[1] ^ hashes[2], 13)
    return np.packbits(p1 ^ p2).tobytes()


def match_bonded_by_boss_db(rdmol: Chem.Mol, hashes: dict, database, cache_bd,
                            cache_ang, cache_dih, cache_imp, missing_bonded, missing_improper):
    params = {}
    impropers = {}
    missing = set()
    missing_i = set()
    for bonded in missing_bonded:
        if len(bonded) == 2:
            bi, bj = bonded
            bond = rdmol.GetBondBetweenAtoms(bi, bj)
            hi = hashes[bi]
            hj = hashes[bj]
            hash_str = _hash_str([hi, hj])
            if cache_bd.get(hash_str) is not None:
                if cache_bd.get(hash_str):
                    ret = cache_bd[hash_str]
                    params[(bi, bj)] = OPLSBond(
                        indices=(bi, bj),
                        k=ret.k,
                        r0=ret.r0,
                        ftype=ret.ftype
                    )
                    logger.debug(f"Found in CACHE for bond (boss database) "
                                 f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()} as"
                                 f" {ret.opls_i}-{ret.opls_j}")
                else:
                    missing.add((bi, bj))
                    logger.debug(f"Found in CACHE for bond (boss database) marked NOT IN DB "
                                 f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()}")
                continue

            res = database.search('bond', hash_str=hash_str)
            if res:
                ret = res[0]
                params[(bi, bj)] = OPLSBond(
                    indices=(bi, bj),
                    k=ret.k,
                    r0=ret.r0,
                    ftype=ret.ftype
                )

                logger.debug(f"Found from boss database for bond "
                             f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()} as"
                             f" {ret.opls_i}-{ret.opls_j}"
                             )
            else:
                missing.add((bi, bj))
                ret = False
                logger.debug(f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()} "
                             f"not found in database")
            cache_bd[hash_str] = copy.deepcopy(ret)

        if len(bonded) == 3:
            angle = bonded
            ai, aj, ak = bonded
            hi, hj, hk = hashes[ai], hashes[aj], hashes[ak]
            hash_str = _hash_str([hi, hj, hk])
            if cache_ang.get(hash_str) is not None:
                if cache_ang.get(hash_str):
                    ret = cache_ang[hash_str]
                    params[angle] = OPLSAngle(
                        indices=angle,
                        k=ret.k,
                        t0=ret.t0,
                        ftype=ret.ftype
                    )
                    logger.debug(f"Found in CACHE for anlge (boss database) "
                                 f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                                 f"{rdmol.GetAtomWithIdx(ak).GetSymbol()} as"
                                 f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}")
                else:
                    missing.add(angle)
                    logger.debug(f"Found in CACHE for anlge (boss database) marked as NOT IN DB "
                                 f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                                 f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}")
                continue  # if hit in cache, continue
            res = database.search('angle', hash_str=hash_str)
            if res:
                ret = res[0]
                params[angle] = OPLSAngle(
                    indices=angle,
                    k=ret.k,
                    t0=ret.t0,
                    ftype=ret.ftype
                )
                logger.debug(f"Found from boss database for angle "
                             f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                             f"{rdmol.GetAtomWithIdx(ak).GetSymbol()} as"
                             f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}")
            else:
                ret = False
                missing.add(angle)
                logger.debug(f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                             f"{rdmol.GetAtomWithIdx(ak).GetSymbol()} "
                             f"not found in database")
            cache_ang[hash_str] = copy.deepcopy(ret)

        if len(bonded) == 4:
            dihedral = bonded
            ai, aj, ak, al = dihedral
            hi, hj, hk, hl = hashes[ai], hashes[aj], hashes[ak], hashes[al]
            hash_str = _hash_str([hi, hj, hk, hl])
            if cache_dih.get(hash_str) is not None:
                if cache_dih[hash_str]:
                    ret = cache_dih[hash_str]
                    params[dihedral] = OPLSDihedral(
                        indices=dihedral,
                        c0=ret.C0,
                        c1=ret.C1,
                        c2=ret.C2,
                        c3=ret.C3,
                        c4=ret.C4,
                        c5=ret.C5,
                        ftype=ret.ftype
                    )
                    logger.debug(f"Found from CACHE for dihedral (boss database) "
                                 f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                                 f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                                 f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
                                 )
                else:
                    missing.add(dihedral)
                    logger.debug(f"Found from CACHE for dihedral (boss database) marked as NOT IN DB "
                                 f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                                 f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()}")
                continue
            res = database.search('dihedral', hash_str=hash_str)
            if res:
                ret = res[0]
                params[dihedral] = OPLSDihedral(
                    indices=dihedral,
                    c0=ret.C0,
                    c1=ret.C1,
                    c2=ret.C2,
                    c3=ret.C3,
                    c4=ret.C4,
                    c5=ret.C5,
                    ftype=ret.ftype
                )
                logger.debug(f"Found from boss database for dihedral "
                             f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                             f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                             f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
                             )
            else:
                ret = False
                missing.add(dihedral)
                logger.debug(f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                             f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} "
                             f"not found in database")
            cache_dih[hash_str] = copy.deepcopy(ret)

    for improper in missing_improper:
        ai, aj, ak, al = improper
        hi, hj, hk, hl = hashes[ai], hashes[aj], hashes[ak], hashes[al]
        hash_str = _hash_str([hi, hj, hk, hl])
        if cache_imp.get(hash_str) is not None:
            if cache_imp[hash_str]:
                ret = cache_imp[hash_str]
                impropers[improper] = OPLSImproper(
                    indices=improper,
                    ftype=4,
                    params=[ret.psi0, ret.k, 2]
                )
                logger.debug(f"Found from CACHE for improper (boss database) "
                             f"{improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                             f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                             f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
                             )
            else:
                missing_i.add(improper)
                logger.debug(f"Found from CACHE for improper (boss database) marked NOT IN DB "
                             f"{improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                             f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()}")
            continue
        res = database.search('improper', hash_str=hash_str)
        if res:
            ret = res[0]
            impropers[improper] = OPLSImproper(
                indices=improper,
                ftype=4,
                params=[ret.psi0, ret.k, 2]
            )
            logger.debug(f"Found from boss database for improper "
                         f"{improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                         f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                         f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
                         )
        else:
            ret = False
            missing_i.add(improper)
            logger.debug(
                f"Improper {improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} "
                f"not found in database")
        cache_imp[hash_str] = copy.deepcopy(ret)
    return params, impropers, missing, missing_i


def match_by_gmx_rule(mol: Chem.Mol, rules=GMXRules):
    rdmol = mol
    params = {}
    _found = set()
    for rule in rules:
        matches = rdmol.GetSubstructMatches(rule.patt)
        if matches:
            for match in matches:
                assert len(match) == 1, ValueError(f"The match should be exactly one atom,"
                                                   f"check your rule smarts {rule.desc}, {rule.smarts}")
                atom_id = match[0]
                atom = rdmol.GetAtomWithIdx(atom_id)
                params[atom_id] = OPLSAtom(opls_num=rule.opls_num,
                                           element=atom.GetSymbol(),
                                           bond_type=rule.bond_type,
                                           mass=rule.mass,
                                           sigma=rule.sigma,
                                           epsilon=rule.epsilon,
                                           charge=rule.charge,
                                           ptype=rule.ptype)
                _found.add(atom_id)
    missing = set(list(range(rdmol.GetNumAtoms()))) - _found
    return params, missing


def match_atom_by_gmx_rule(mol: Chem.Mol, ob_mol: ob.OBMol, hashes: dict, cache: dict, rules=GMXRules):
    rdmol = mol
    params = {}
    _found = set()
    for atom in rdmol.GetAtoms():
        _hash_str = np.packbits(hashes[atom.GetIdx()]).tobytes()
        if cache.get(_hash_str) is not None:
            if cache.get(_hash_str):
                params[atom.GetIdx()] = cache.get(_hash_str)
                logger.debug(f"Found from cache for atom {atom.GetIdx()}: "
                             f"{atom.GetSymbol()}, {cache.get(_hash_str).bond_type}")
                _found.add(atom.GetIdx())
            else:
                logger.debug(f"Atom {atom.GetIdx()} is in cache and marked with NOT FOUND FLAG")
            continue
        ob_atom: ob.OBAtom = ob_mol.GetAtomById(atom.GetIdx())
        for rule in rules:
            if ob_atom.MatchesSMARTS(rule.smarts):
                if logger.level <= logging.DEBUG:
                    stat = _get_stat(rdmol, atom.GetIdx())
                    logger.debug(f"The center atom {atom.GetIdx()}, {atom.GetSymbol()},"
                                 f"{stat} matches {rule.desc}")
                opls_atom = OPLSAtom(opls_num=rule.opls_num,
                                     element=atom.GetSymbol(),
                                     bond_type=rule.bond_type,
                                     mass=rule.mass,
                                     sigma=rule.sigma,
                                     epsilon=rule.epsilon,
                                     charge=rule.charge,
                                     ptype=rule.ptype)
                params[atom.GetIdx()] = opls_atom
                _found.add(atom.GetIdx())
                cache[_hash_str] = copy.deepcopy(opls_atom)
                break
        else:
            # not found flag
            cache[_hash_str] = False
    missing = set(list(range(rdmol.GetNumAtoms()))) - _found
    return params, missing


def match_bonded_by_gmx_rule(params_atoms, bond_idx, angle_idx, dihedral_idx):
    params = {}
    missing = set()
    with open(os.path.join(__this_dir__, 'resources', 'gmx', 'bonded_rules.dat'), 'rb') as f:
        rules = pickle.load(f)
    for bond in bond_idx:
        bi, bj = bond
        oi, oj = params_atoms.get(bi), params_atoms.get(bj)
        if not (oi and oj):
            missing.add((bi, bj))
            logger.debug(f"The bond {bi}: {oi} and {bj}: {oj} contain missing types.")
            continue
        _name = f"{oi.bond_type}-{oj.bond_type}||{oj.bond_type}-{oi.bond_type}"
        rule = rules.get(_name)
        if rule:
            params[(bi, bj)] = OPLSBond(indices=(bi, bj), k=rule.k, r0=rule.r0, ftype=rule.ftype)
            logger.debug(f"Found {rule.atom_types} for {(bi, bj)}")
        else:
            logger.debug(f"Cant find bond {oi.bond_type}-{oj.bond_type}")
            missing.add((bi, bj))

    for angle in angle_idx:
        ai, aj, ak = angle
        oi, oj, ok = params_atoms.get(ai), params_atoms.get(aj), params_atoms.get(ak)
        if not (oi and oj and ok):
            logger.debug(f"The angle {ai}: {oi}, {aj}: {oj} and {ak}: {ok} contain missing types.")
            missing.add(angle)
            continue
        _name = f"{oi.bond_type}-{oj.bond_type}-{ok.bond_type}"
        rule = rules.get(_name)
        if not rule:
            _name = f"{ok.bond_type}-{oj.bond_type}-{oi.bond_type}"
            rule = rules.get(_name)

        if rule:
            params[(ai, aj, ak)] = OPLSAngle(indices=(ai, aj, ak), k=rule.k, t0=rule.t0, ftype=rule.ftype)
            logger.debug(f"Found {rule.atom_types} for {(ai, aj, ak)}")
        else:
            missing.add((ai, aj, ak))
            logger.debug(f"Cant find angle {oi.bond_type}-{oj.bond_type}-{ok.bond_type}")

    for dihedral in dihedral_idx:
        ai, aj, ak, al = dihedral
        oi, oj, ok, ol = params_atoms.get(ai), params_atoms.get(aj), params_atoms.get(ak), params_atoms.get(al)
        if not (oi and oj and ok and ol):
            logger.debug(f"The dihedral {ai}: {oi}, {aj}: {oj}, {ak}: {ok}, {al}: {ol} contain missing types.")
            missing.add(dihedral)
            continue
        _name = f"{oi.bond_type}-{oj.bond_type}-{ok.bond_type}-{ol.bond_type}"
        rule = rules.get(_name)
        if not rule:
            _name = f"{ol.bond_type}-{ok.bond_type}-{oj.bond_type}-{oi.bond_type}"  # ABCD DCBA
            rule = rules.get(_name)

        if not rule:
            _name = f"X-{oj.bond_type}-{ok.bond_type}-X"
            rule = rules.get(_name)

        if not rule:
            _name = f"X-{ok.bond_type}-{oj.bond_type}-X"  # X-BC-X X-CB-X
            rule = rules.get(_name)

        if not rule:
            _name = f"{oi.bond_type}-{oj.bond_type}-{ok.bond_type}-X"
            rule = rules.get(_name)

        if not rule:
            _name = f"{ol.bond_type}-{ok.bond_type}-{oj.bond_type}-X"  # ABC-X DCB-X
            rule = rules.get(_name)

        if not rule:
            _name = f"{oi.element}-{oj.bond_type}-{ok.bond_type}-{ol.bond_type}"
            rule = rules.get(_name)

        if not rule:
            _name = f"{ol.element}-{ok.bond_type}-{oj.bond_type}-{oi.bond_type}"  # ?BCD ?CBA
            rule = rules.get(_name)

        if not rule:
            _name = f"{oi.bond_type}-{oj.bond_type}-{ok.bond_type}-{ol.element}"
            rule = rules.get(_name)

        if not rule:
            _name = f"{ol.bond_type}-{ok.bond_type}-{oj.bond_type}-{oi.element}"  # ABC? DCB?
            rule = rules.get(_name)

        if not rule:
            _name = f"{oi.element}-{oj.bond_type}-{ok.bond_type}-{ol.element}"
            rule = rules.get(_name)

        if not rule:
            _name = f"{ol.element}-{ok.bond_type}-{oj.bond_type}-{oi.element}"  # ?BC? ?CB?
            rule = rules.get(_name)

        if rule:
            params[(ai, aj, ak, al)] = OPLSDihedral(indices=(ai, aj, ak, al), ftype=rule.ftype, c0=rule.c0,
                                                    c1=rule.c1,
                                                    c2=rule.c2, c3=rule.c3, c4=rule.c4, c5=rule.c5)
            logger.debug(f"Mathing rule with {rule.atom_types} for dihedral {dihedral},"
                         f"{oi.bond_type}-{oj.bond_type}-{ok.bond_type}-{ol.bond_type}")
        else:
            logger.debug(f"Cant find dihedral {oi.bond_type}-{oj.bond_type}-{ok.bond_type}-{ol.bond_type}")
            missing.add((ai, aj, ak, al))
    return params, missing


def match_improper_by_gmx_rule(params_atoms, improper_idx):
    missing = set()
    params = {}
    for improper in improper_idx:
        ai, aj, ak, al = improper  # aj is the center atom
        oi, oj, ok, ol = params_atoms.get(ai), params_atoms.get(aj), params_atoms.get(ak), params_atoms.get(al)
        if not (oi and oj and ok and ol):
            logger.debug(f"The improper {ai}: {oi}, {aj}: {oj}, {ak}: {ok}, {al}: {ol} contain missing types.")
            missing.add(improper)
            continue

        perms = list(permutations([ai, ak, al]))
        ret = None
        for _perm in perms:
            i = _perm[0]
            k = _perm[1]
            l = _perm[2]
            oi_, ok_, ol_ = params_atoms.get(i), params_atoms.get(k), params_atoms.get(l)
            dihedral = [oi_.bond_type, oj.bond_type,
                        ok_.bond_type, ol_.bond_type]
            d_str = '-'.join(dihedral)
            if re.search(r'O.*-(C|C_2|C_3)-.*-.*', d_str):
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 43.93200, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break
            elif re.search(r'.*-NO-ON-NO', d_str):
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 43.93200, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break
            elif re.search(r'N2-.*-N2-N2', d_str):
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 43.93200, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break
            elif re.search(r'.*-N.+-.*-.*', d_str):
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 4.18400, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break
            elif re.search(r'.*-N-.*-.*', d_str):  # oplsaa.par
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 10.46, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break
            elif re.search(r'.*-(CM|C=)-.*-.*', d_str):
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 62.76000, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break
            elif re.search(r'.*-(CM|CB|CN|CV|CW|CR|CK|CQ|CS|C*)-.*-.*', d_str):
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 4.60240, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break
            elif re.search(r'.*-(CA)-.*-.*', d_str):  # oplsaa.par
                ret = OPLSImproper(indices=(i, aj, k, l), ftype=4, params=[180.0, 10.46, 2])
                logger.debug(f"Mathing improper {improper} for {d_str} with order {(i, aj, k, l)}")
                break

        if not ret:
            missing.add(improper)
            logger.debug(f"Mathing improper for"
                         f"{improper}: {oi.bond_type}-{oj.bond_type}-{ok.bond_type}-{ol.bond_type} failed.")
        else:
            params[ret.indices] = ret
    return params, missing
