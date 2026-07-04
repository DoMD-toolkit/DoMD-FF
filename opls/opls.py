from openbabel import openbabel as ob

from rdkit import Chem

from lib import get_opls_bonded_idx, count_bonded
from misc.logger import logger
from opls._misc import OPLSAtom
from opls.functions import (
    _build_hash,
    match_atom_by_gmx_rule,
    match_by_gmx_rule,
    match_bonded_by_gmx_rule,
    match_improper_by_gmx_rule,
    match_atom_by_boss_db,
    match_bonded_by_boss_db,
    match_params_ml
)
from opls.opls_db import opls_db

# opls_406   Li+  3   6.94100     1.000       A    2.12645e-01  7.64793e-02
# opls_404   Li+  3   6.94100     1.000       A    1.25992e-01  2.61500e+01
# opls_407   Na+  11  22.98977     1.000       A    3.33045e-01  1.15980e-02
# opls_405   Na+  11  22.98977     1.000       A    1.89744e-01  6.72427e+00

ION_TBL = {
    'Li': {
        'aq': OPLSAtom(opls_num=406, bond_type='Li+', element='Li', mass=6.941,
                       sigma=2.12645e-01, epsilon=7.64793e-02, charge=1.0, ptype='A'),
        'cond': OPLSAtom(opls_num=404, bond_type='Li+', element='Li', mass=6.941,
                       sigma=1.25992e-01, epsilon=2.61500e+01, charge=1.0, ptype='A')
    },
    'Na': {
        'aq': OPLSAtom(opls_num=407, bond_type='Na+', element='Na', mass=22.990,
                       sigma=3.33045e-01, epsilon=1.15980e-02, charge=1.0, ptype='A'),
        'cond': OPLSAtom(opls_num=405, bond_type='Na+', element='Na', mass=22.990,
                       sigma=1.89744e-01, epsilon=6.72427e+00, charge=1.0, ptype='A')
    },
}

THRESHOLD_L = 5000
THRESHOLD_H = 5000


def opls_setup(rdmol: Chem.Mol, obmol: ob.OBMol = None, useGMX=True,
               useBOSS=False, useML=False, overwrite=False, ion_env='aq'):
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
    if n_atoms > THRESHOLD_H and useGMX:
        logger.warn(f"The target molecule has more than {THRESHOLD_H} atoms, template method is not available."
                    f"I'll set `useGMX=False`.")
        useGMX = False
    if useML:
        # 1. Base check for total atom count
        if n_atoms < 4:
            useML = False
        else:
            # 2. Advanced check: ensure at least one sequential 4-atom chain exists (3 contiguous bonds)
            # length=3 with useBonds=True searches for paths consisting of exactly 3 sequential bonds (A-B-C-D)
            has_dihedral = len(Chem.FindAllPathsOfLengthN(rdmol, 3, useBonds=True)) > 0

            if not has_dihedral:
                logger.warning(
                    f"The molecule contains {n_atoms} atoms but lacks a sequential 4-atom pathway. "
                    f"No chemical dihedral angle can be defined. Disabling useML."
                )
                useML = False
    params_atoms = {}
    params_bonded = {}
    params_impropers = {}

    _cache_gmx = {}
    _cache_boss = {}
    _cache_boss_bd = {}
    _cache_boss_ang = {}
    _cache_boss_dih = {}
    _cache_boss_imp = {}

    bond_idx, angle_idx, dihedral_idx, improper_idx = get_opls_bonded_idx(rdmol)
    # all missing
    missing_atoms = set(list(range(rdmol.GetNumAtoms())))
    missing_bonded = set.union(set(bond_idx), set(angle_idx), set(dihedral_idx))
    missing_impropers = set(improper_idx)

    logger.info(f"Overwrite mode is {overwrite}, if `overwrite=True`, the each method will find all parameters "
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
            logger.warn(f"(GMX finder) Missing/Total {len(missing_gmx_atoms)}/{rdmol.GetNumAtoms()} "
                        f"atom types for GMX template search!")
        if not overwrite:
            # if overwrite=False, the next method only finds the missing of previous methods
            # else the missing_xxx keep as initialized
            missing_atoms = missing_atoms.intersection(missing_gmx_atoms)

        params_atoms.update(opls_gmx_atoms)
        opls_gmx_bonded, missing_gmx_bonded = match_bonded_by_gmx_rule(params_atoms, bond_idx, angle_idx, dihedral_idx)
        params_bonded.update(opls_gmx_bonded)

        if len(missing_gmx_bonded) > 0:
            m_b, m_a, m_d = count_bonded(missing_bonded)
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

        m_b, m_a, m_d = count_bonded(opls_gmx_bonded)
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
            m_b, m_a, m_d = count_bonded(missing_boss_bonded)
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

        m_b, m_a, m_d = count_bonded(opls_boss_bonded)
        logger.info(f"BOSS searching total found {len(opls_boss_atoms)}/{rdmol.GetNumAtoms()} atoms, "
                    f"{m_b}/{len(bond_idx)}, {m_a}/{len(angle_idx)}, {m_d}/{len(dihedral_idx)} bonds, angles, "
                    f"dihedrals, and {len(opls_boss_improper)}/{len(improper_idx)} impropers.")

    if useML:
        # find missing only
        opls_ml_atoms, opls_ml_bonded, opls_ml_improper = match_params_ml(rdmol,
                                                                          missing_atoms,
                                                                          missing_bonded,
                                                                          missing_impropers)
        params_atoms.update(opls_ml_atoms)
        params_bonded.update(opls_ml_bonded)
        params_impropers.update(opls_ml_improper)
    logger.info(f"Total Found atoms/Total atoms: {len(params_atoms)}/{rdmol.GetNumAtoms()}")
    m_b, m_a, m_d = count_bonded(params_bonded)
    logger.info(f"Found bonds/Total angles/Total dihedrals/Total impropers/Total: {m_b}/{len(bond_idx)}"
                f" {m_a}/{len(angle_idx)} {m_d}/{len(dihedral_idx)} {len(params_impropers)}/{len(improper_idx)}")

    success = (len(params_atoms) == rdmol.GetNumAtoms() and m_b == len(bond_idx) and m_a == len(
        angle_idx) and m_d == len(dihedral_idx) and len(params_impropers) == len(improper_idx))

    meta = {'n_atom': len(params_atoms), 'n_bond': m_b, 'n_ang': m_a, 'n_dih': m_d, 'n_imp': len(params_impropers),
            't_atom': rdmol.GetNumAtoms(), 't_bond': len(bond_idx), 't_ang': len(angle_idx), 't_dih': len(dihedral_idx),
            't_imp': len(params_impropers)}

    for atom_idx in params_atoms:
        atom = params_atoms[atom_idx]
        if ION_TBL.get(atom.element) is not None:
            ion = ION_TBL.get(atom.element).get(ion_env)
            if ion is not None:
                params_atoms[atom_idx] = ion

    return ((params_atoms, params_bonded, params_impropers),
            (missing_atoms, missing_bonded, missing_impropers),
            success,
            meta)
