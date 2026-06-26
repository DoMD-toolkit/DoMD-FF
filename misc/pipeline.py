import os
import sys
import logging
import numpy as np
from rdkit import Chem

from ForceField import FF
from misc.logger import logger
from misc.io.gmx import write_gro_file, write_top_file, write_itp_file, write_atomtypes_head, \
    write_top_file_with_includes
#logger.setLevel('ERROR')

def _extract_mol_metadata(mol):
    """Helper function to extract structural properties directly from an RDKit molecule object."""
    num_atoms = mol.GetNumAtoms()
    coordinates = mol.GetConformer().GetPositions()

    res_names = mol.GetProp("RES_NAMES").split() if mol.HasProp("RES_NAMES") else ["UNL"] * num_atoms
    res_ids = [int(x) for x in mol.GetProp("RES_NUMS").split()] if mol.HasProp("RES_NUMS") else [1] * num_atoms

    if mol.HasProp("BOX_TENSOR"):
        box_tensor = [float(x) for x in mol.GetProp("BOX_TENSOR").split()]
    else:
        max_coords = np.max(coordinates, axis=0) if num_atoms > 0 else np.array([50.0, 50.0, 50.0])
        min_coords = np.min(coordinates, axis=0) if num_atoms > 0 else np.array([0.0, 0.0, 0.0])
        dx, dy, dz = max_coords - min_coords + 5.0
        box_tensor = [dx, dy, dz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return coordinates, res_names, res_ids, box_tensor


def run_itp_mode(mols, output_dir, obmols=None, molecule_name=None):
    """
    Case 1: itp mode
    Receives a Python list of single RDKit molecule objects.
    Strictly validates the single-fragment connectivity of each item.
    Outputs independent .itp and .gro files along with integrated master files.
    """
    logger.info("===> Starting itp mode from list of RDKit molecule objects")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not mols:
        logger.error("The provided molecule list is empty.")
        return

    num_mols = len(mols)
    logger.info(f"Detected {num_mols} molecule object(s) in the input list.")

    # 1. Strict connectivity verification
    for idx, mol in enumerate(mols):
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            error_msg = (
                f"\n[Input Error]: In itp mode, each molecule object must be a strictly connected single molecule.\n"
                f"Detected that molecule object at index {idx} contains multiple disconnected fragments.\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 2. Characterize and parameterize each molecule object purely in-memory
    # head_itp_out = os.path.join(output_dir, f"{base_name}_atomtypes.itp")
    # top_out = os.path.join(output_dir, f"{base_name}.top")
    forcefields_meta = {}
    coordinates_meta = {}
    if molecule_name is None:
        molecule_name = {}
        for idx in range(num_mols):
            molecule_name[idx] = f'MOL_{idx:0>4d}'

    for idx, mol in enumerate(mols):
        gro_out = os.path.join(output_dir, f"{molecule_name[idx]}.gro")
        itp_out = os.path.join(output_dir, f"{molecule_name[idx]}.itp")

        # Extract values directly from memory object
        coordinates, res_names, res_ids, box_tensor = _extract_mol_metadata(mol)

        # Retrieve mapped openbabel instance dynamically if supplied
        obmol = None
        if obmols is not None:
            if isinstance(obmols, dict):
                obmol = obmols.get(idx)
            elif isinstance(obmols, list) and idx < len(obmols):
                obmol = obmols[idx]
        forcefield = FF('opls')
        forcefield.setup(mol, obmol, useGMX=True, useBOSS=True, overwrite=False, useML=True)

        coordinates_meta[idx] = (coordinates, res_names, res_ids, box_tensor, gro_out)
        forcefields_meta[idx] = (forcefield, res_names, res_ids, itp_out)

    # 3. Deduplicate atomtypes globally across all structures
    params_atom_all = {}
    global_atom_idx = 0
    for idx in forcefields_meta:
        forcefield, _, _, _ = forcefields_meta[idx]
        params_atom, params_bonded, params_improper = forcefield.params
        for atom_idx in params_atom:
            params_atom_all[global_atom_idx] = params_atom[atom_idx]
            global_atom_idx += 1
    # unique_atomtypes, type2name = map_unique_atomtypes(params_atom_all)
    # write_atomtypes_head(head_itp_out, unique_atomtypes)

    itp_files_name = {}
    mol_counts = {}
    for idx in coordinates_meta:
        coordinates, res_names, res_ids, box_tensor, gro_out = coordinates_meta[idx]
        forcefield, res_names, res_ids, itp_out = forcefields_meta[idx]
        params_atom, params_bonded, params_improper = forcefield.params
        mol_name = molecule_name[idx]
        atom_names = [params_atom[i].element for i in range(len(params_atom))]
        write_gro_file(gro_out, coordinates, res_names, res_ids, box_tensor, atom_names)
        write_itp_file(itp_out, forcefield, res_names, res_ids,mol_name=mol_name,write_atomtypes=True)

        logger.info(f"Successfully generated files for molecule object {idx + 1}: {gro_out} & {itp_out}")
        itp_files_name[idx] = itp_out
        mol_counts[idx] = 1

    logger.info("===> itp mode finished successfully.\n")


def run_top_mode(rdmol, output_dir, base_name="system", obmol=None):
    """
    Case 2: top mode
    Receives a single integrated RDKit molecule object representing the entire system.
    Bypasses connectivity constraints, supporting multi-fragment architectures directly.
    Outputs a standalone monolithic total system .top and .gro file block.
    """
    logger.info("===> Starting top mode from single system RDKit molecule object")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if rdmol is None:
        logger.error("The provided system RDMol object is invalid (None).")
        return
    else:
        Chem.SanitizeMol(rdmol)

    gro_out = os.path.join(output_dir, f"{base_name}.gro")
    top_out = os.path.join(output_dir, f"{base_name}.top")

    # Extract all required geometry and metadata completely in-memory
    coordinates, res_names, res_ids, box_tensor = _extract_mol_metadata(rdmol)

    # Setup force field mapping with provided optional OBMol instance
    forcefield = FF('opls')
    forcefield.setup(rdmol, obmol, useGMX=True, useBOSS=True, overwrite=False, useML=True)
    params_atom, params_bonded, params_improper = forcefield.params
    atom_names = [params_atom[i].element for i in range(len(params_atom))]

    # Directly export consolidated structural records
    write_gro_file(gro_out, coordinates, res_names, res_ids, box_tensor, atom_names)
    write_top_file(top_out, forcefield, res_names, res_ids)

    logger.info(f"Successfully generated total system files: {gro_out} & {top_out}")
    logger.info("===> top mode finished successfully.\n")


