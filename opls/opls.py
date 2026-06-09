from itertools import combinations
from openbabel import openbabel as ob

from rdkit import Chem

from misc.logger import logger
from opls.functions import (
    _build_hash,
    match_atom_by_gmx_rule,
    match_by_gmx_rule,
    match_bonded_by_gmx_rule,
    match_improper_by_gmx_rule,
    match_atom_by_boss_db,
    match_bonded_by_boss_db,
)
from opls.opls_db import opls_db

THRESHOLD_L = 5000
THRESHOLD_H = 20000


def _count_bonded(bonded):
    m_b = m_a = m_d = 0
    for m in bonded:
        if len(m) == 2:
            m_b += 1
        if len(m) == 3:
            m_a += 1
        if len(m) == 4:
            m_d += 1
    return m_b, m_a, m_d


def _get_opls_bonded_idx(rdmol: Chem.Mol):
    bond_idx, angle_idx, dihedral_idx, improper_idx = set(), set(), set(), set()
    for bond in rdmol.GetBonds():
        bond_idx.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_i, atom_j = rdmol.GetAtomWithIdx(bi), rdmol.GetAtomWithIdx(bj)
        for atom_k in atom_i.GetNeighbors():
            bk = atom_k.GetIdx()
            if bk == bj:
                continue
            for atom_l in atom_k.GetNeighbors():
                bl = atom_l.GetIdx()
                if bl in (bi, bj, bk):
                    continue
                tpl = (bk, bi, bj, bl)
                if tpl not in bond_idx:
                    dihedral_idx.add(tpl)
    for atom in rdmol.GetAtoms():
        j = atom.GetIdx()
        nbrs = [_.GetIdx() for _ in atom.GetNeighbors()]
        for i, k in combinations(nbrs, 2):
            tpl = (i, j, k)
            if tpl not in angle_idx:
                angle_idx.add(tpl)
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        if len(atom.GetNeighbors()) == 3 and atom.GetHybridization().name == 'SP2':
            neighbors = list(atom.GetNeighbors())
            # the center atom is always at j.
            i, j, k, l = neighbors[0].GetIdx(), idx, neighbors[1].GetIdx(), neighbors[2].GetIdx()
            improper_idx.add((i, j, k, l))
    return bond_idx, angle_idx, dihedral_idx, improper_idx


def opls_setup(rdmol: Chem.Mol, obmol: ob.OBMol = None, useGMX=True, useBOSS=False, useML=False, overwrite=False):
    n_atoms = rdmol.GetNumAtoms()
    ob_success = obmol is not None
    if obmol is None and n_atoms > THRESHOLD_L:
        sdf_block = Chem.MolToMolBlock(rdmol, forceV3000=True)
        ob_conv = ob.OBConversion()
        ob_conv.SetInFormat("sdf")
        obmol = ob.OBMol()
        ob_success = ob_conv.ReadString(obmol, sdf_block)
        if not ob_success:
            logger.warn(f"The target molecule has more than {THRESHOLD_L} atoms, "
                        f"but I can't turn it into an OBMol, the template searching "
                        f"will be performed with rdkit, which may be extremely slow.")
    if n_atoms > THRESHOLD_H:
        logger.warn(f"The target molecule has more than {THRESHOLD_H} atoms, template method is not available."
                    f"I'll set `useGMX=False`.")
        useGMX = False

    params_atoms = {}
    params_bonded = {}
    params_impropers = {}

    _cache_gmx = {}
    _cache_boss = {}
    _cache_boss_bd = {}
    _cache_boss_ang = {}
    _cache_boss_dih = {}
    _cache_boss_imp = {}

    bond_idx, angle_idx, dihedral_idx, improper_idx = _get_opls_bonded_idx(rdmol)
    missing_atoms = set(list(range(rdmol.GetNumAtoms())))
    missing_bonded = set.union(set(bond_idx), set(angle_idx), set(dihedral_idx))
    missing_impropers = set(improper_idx)

    logger.warn(f"Overwrite mode is {overwrite}, if `overwrite=True`, the each method will find all parameters "
                f"(GMX->BOSS->ML) independently, and overwrites existing matches of previous methods. "
                f"If `overwrite=False`, the next method will only try to find missing types of the former methods.")

    atom_hashes = {}
    if useGMX or useBOSS:
        logger.info("Building atom hashes for GMX and BOSS searching methods.")
        atom_hashes, mol_hash = _build_hash(rdmol)

    if useGMX:
        if ob_success:
            opls_gmx_atoms, missing_gmx_atoms = match_atom_by_gmx_rule(rdmol,
                                                                       obmol,
                                                                       atom_hashes,
                                                                       _cache_gmx)
        else:
            opls_gmx_atoms, missing_gmx_atoms = match_by_gmx_rule(rdmol)

        if len(missing_gmx_atoms) > 0:
            logger.warn(f"(GMX finder) Missing/Total {len(missing_gmx_atoms)}/{rdmol.GetNumAtoms} "
                        f"atom types for GMX template search!")
        if not overwrite:
            missing_atoms = missing_atoms.intersection(missing_gmx_atoms)

        params_atoms.update(opls_gmx_atoms)
        opls_gmx_bonded, missing_gmx_bonded = match_bonded_by_gmx_rule(params_atoms, bond_idx, angle_idx, dihedral_idx)
        params_bonded.update(opls_gmx_bonded)

        if len(missing_gmx_bonded) > 0:
            m_b, m_a, m_d = _count_bonded(missing_bonded)
            logger.warn(f"(GMX finder) Missing/Total "
                        f"{m_b}/{len(bond_idx)}, {m_a}/{len(angle_idx)}, {m_d}/{len(dihedral_idx)} "
                        f"bond, angle, dihedral types for GMX template search!")

        if not overwrite:
            missing_bonded = missing_bonded.intersection(missing_gmx_bonded)
        opls_gmx_improper, missing_gmx_improper = match_improper_by_gmx_rule(params_atoms, improper_idx)

        if len(missing_gmx_improper) > 0:
            logger.warn(f"(GMX finder) Missing/Total {len(missing_gmx_improper)}/{len(improper_idx)} "
                        f"improper types for GMX template search!")

        if not overwrite:
            missing_impropers = missing_impropers.intersection(missing_gmx_improper)
        params_impropers.update(opls_gmx_improper)

        m_b, m_a, m_d = _count_bonded(opls_gmx_bonded)
        logger.info(f"GMX searching total found {len(opls_gmx_atoms)}/{rdmol.GetNumAtoms()} atoms, "
                    f"{m_b}/{len(bond_idx)}, {m_a}/{len(angle_idx)}, {m_d}/{len(dihedral_idx)} bonds, angles, "
                    f"dihedrals, and {len(opls_gmx_improper)}/{len(improper_idx)} impropers.")

    if useBOSS:
        opls_boss_atoms, missing_boss_atoms = match_atom_by_boss_db(rdmol, atom_hashes,
                                                                    opls_db, _cache_boss,
                                                                    missing_atoms)
        if len(missing_boss_atoms) > 0:
            logger.warn(f"(BOSS finder) Missing/Total Missing/Total "
                        f"{len(missing_boss_atoms)}/{len(missing_atoms)}/{rdmol.GetNumAtoms()}")
        else:
            logger.info(f"(BOSS finder) Found all missing/total {len(missing_atoms)}/{rdmol.GetNumAtoms()} atoms.")

        if not overwrite:
            missing_atoms = missing_atoms.intersection(missing_boss_atoms)
        params_atoms.update(opls_boss_atoms)

        opls_boss_bonded, opls_boss_improper, missing_boss_bonded, missing_boss_improper = match_bonded_by_boss_db(
            rdmol,
            atom_hashes,
            opls_db,
            _cache_boss_bd,
            _cache_boss_ang,
            _cache_boss_dih,
            _cache_boss_imp,
            missing_bonded,
            missing_impropers
        )
        if len(missing_boss_bonded) > 0:
            m_b, m_a, m_d = _count_bonded(missing_boss_bonded)
            logger.warn(f"(BOSS finder) Missing/Total "
                        f"{m_b}/{len(bond_idx)}, {m_a}/{len(angle_idx)}, {m_d}/{len(dihedral_idx)} "
                        f"bond, angle, dihedral types for BOSS search!")

        if not overwrite:
            missing_bonded = missing_bonded.intersection(missing_boss_bonded)

        if len(missing_boss_improper) > 0:
            logger.warn(f"(BOSS finder) Missing/Total {len(missing_boss_improper)}/{len(improper_idx)} "
                        f"improper types for BOSS search!")

        if not overwrite:
            missing_impropers = missing_impropers.intersection(missing_boss_improper)

        params_bonded.update(opls_boss_bonded)
        params_impropers.update(opls_boss_improper)

        m_b, m_a, m_d = _count_bonded(opls_boss_bonded)
        logger.info(f"BOSS searching total found {len(opls_boss_atoms)}/{rdmol.GetNumAtoms()} atoms, "
                    f"{m_b}/{len(bond_idx)}, {m_a}/{len(angle_idx)}, {m_d}/{len(dihedral_idx)} bonds, angles, "
                    f"dihedrals, and {len(opls_boss_improper)}/{len(improper_idx)} impropers.")

    if useML:
        pass

    logger.info(f"Total Found atoms/Total atoms: {len(params_atoms)}/{rdmol.GetNumAtoms()}")
    m_b, m_a, m_d = _count_bonded(params_bonded)
    logger.info(f"Found bonds/Total angles/Total dihedrals/total impropers/total: {m_b}/{len(bond_idx)}"
                f" {m_a}/{len(angle_idx)} {m_d}/{len(dihedral_idx)} {len(params_impropers)}/{len(improper_idx)}")

    success = (len(params_atoms) == rdmol.GetNumAtoms() and m_b == len(bond_idx) and m_a == len(
        angle_idx) and m_d == len(dihedral_idx) and len(params_impropers) == len(improper_idx))

    return (params_atoms, params_bonded, params_impropers), (missing_atoms, missing_bonded, missing_impropers), success
