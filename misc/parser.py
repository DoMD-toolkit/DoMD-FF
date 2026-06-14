import os

import numpy as np
from rdkit import Chem
from openbabel import openbabel as ob

from misc.logger import logger


def sdf_load_all_as_one(input_path):
    suppl = Chem.SDMolSupplier(input_path, removeHs=False)
    rd_combined_mol = None
    for mol in suppl:
        if mol is None: continue
        if rd_combined_mol is None:
            rd_combined_mol = mol
        else:
            rd_combined_mol = Chem.CombineMols(rd_combined_mol, mol)
    ob_combined_mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("sdf")
    ob_mol = ob.OBMol()
    notatend = obConversion.ReadFile(ob_mol, input_path)

    while notatend:
        ob_combined_mol += ob_mol
        ob_mol = ob.OBMol()
        notatend = obConversion.Read(ob_mol)

    success = rd_combined_mol.GetNumAtoms() == ob_combined_mol.NumAtoms()
    return rd_combined_mol, ob_combined_mol, success


def molecule_reader(input_path):
    """
    工业级大体系读取器
    支持 PDB 和 SDF (V3000) 格式。
    具备残基编号溢出自动还原（Unwrap）机制与全自动默认兜底。

    返回:
        obmol: Ob mol
        rdmol: RDKit 分子对象
        coordinates: Nx3 的 numpy 数组 (单位: 埃)
        res_names: 长度为 N 的列表 (每个原子的残基名)
        res_ids: 长度为 N 的列表 (每个原子的真实大残基编号，可达百万)
        box_tensor: 长度为 9 的列表 (埃)
    """
    ext = os.path.splitext(input_path)[-1].lower().replace('.', '')


    if ext == 'pdb':
        rdmol = Chem.MolFromPDBFile(input_path, removeHs=False)
        obmol = ob.OBMol()
        obconv = ob.OBConversion()
        obconv.SetInFormat('pdb')
        ob_suc = obconv.ReadFile(obmol, input_path)
    elif ext == 'sdf':
        rdmol, obmol, ob_suc = sdf_load_all_as_one(input_path)
    else:
        raise ValueError("Only PDB and SDF files are supported!")


    if not rdmol or not rdmol.GetNumConformers() or not ob_suc:
        raise ValueError("Read file failed!")

    conf = rdmol.GetConformer()
    num_atoms = rdmol.GetNumAtoms()
    coordinates = conf.GetPositions()  # Nx3 numpy array

    res_names = []
    res_ids = []
    box_tensor = [0.0] * 9
    has_box = False
    has_res_info = False

    if ext == 'pdb':
        with open(input_path, 'r') as f:
            for line in f:
                if line.startswith("CRYST1"):
                    try:
                        a = float(line[6:15])
                        b = float(line[15:24])
                        c = float(line[24:33])
                        box_tensor = [a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c]
                        has_box = True
                    except ValueError:
                        pass
                    break

        last_file_rid = -1
        wrap_counter = 0

        first_atom_info = rdmol.GetAtomWithIdx(0).GetMonomerInfo()
        if first_atom_info:
            has_res_info = True
            for atom in rdmol.GetAtoms():
                info = atom.GetMonomerInfo()
                r_name = info.GetResidueName().strip()
                current_file_rid = info.GetResidueNumber()

                if last_file_rid != -1 and current_file_rid < last_file_rid:
                    wrap_counter += 1

                true_rid = current_file_rid + (wrap_counter * 10000)

                res_names.append(r_name if r_name else "RES")
                res_ids.append(true_rid)
                last_file_rid = current_file_rid


    elif ext == 'sdf':
        if rdmol.HasProp("BOX_TENSOR") and rdmol.HasProp("RES_NAMES") and rdmol.HasProp("RES_NUMS"):
            try:
                box_tensor = [float(x) for x in rdmol.GetProp("BOX_TENSOR").split()]
                res_names = rdmol.GetProp("RES_NAMES").split()
                res_ids = [int(x) for x in rdmol.GetProp("RES_NUMS").split()]

                if len(box_tensor) == 9 and len(res_names) == num_atoms and len(res_ids) == num_atoms:
                    has_box = True
                    has_res_info = True
            except ValueError:
                pass

    if not has_res_info:
        logger.warn("No residue info detected, all residues will be named as 'UNL'，with id=1")
        res_names = ["UNL"] * num_atoms
        res_ids = [1] * num_atoms

    if not has_box:
        logger.warn("No box info detected, the box is set to be +5A of max - min")
        max_coords = np.max(coordinates, axis=0)
        min_coords = np.min(coordinates, axis=0)
        dx, dy, dz = max_coords - min_coords + 5.0
        box_tensor = [dx, 0.0, 0.0, 0.0, dy, 0.0, 0.0, 0.0, dz]

    return obmol, rdmol, coordinates, res_names, res_ids, box_tensor
