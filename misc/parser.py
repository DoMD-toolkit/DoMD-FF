import os

import numpy as np
from rdkit import Chem
from openbabel import openbabel as ob

from misc.logger import logger


def load_all_as_one_rdmol(input_path):
    suppl = Chem.SDMolSupplier(input_path, removeHs=False)
    combined_mol = None
    for mol in suppl:
        if mol is None: continue
        if combined_mol is None:
            combined_mol = mol
        else:
            combined_mol = Chem.CombineMols(combined_mol, mol)

    return combined_mol

def molecule_reader(input_path):
    """
    工业级大体系读取器
    支持 PDB 和 SDF (V3000) 格式。
    具备残基编号溢出自动还原（Unwrap）机制与全自动默认兜底。

    返回:
        mol: RDKit 分子对象
        coordinates: Nx3 的 numpy 数组 (单位: 埃)
        res_names: 长度为 N 的列表 (每个原子的残基名)
        res_ids: 长度为 N 的列表 (每个原子的真实大残基编号，可达百万)
        box_tensor: 长度为 9 的列表 (埃)
    """
    ext = os.path.splitext(input_path)[-1].lower().replace('.', '')
    obmol = ob.OBMol()
    obconv = ob.OBConversion()

    # 1. 使用 RDKit 读取基础分子和坐标
    if ext == 'pdb':
        mol = Chem.MolFromPDBFile(input_path, removeHs=False)
        obconv.SetInFormat('pdb')
    elif ext == 'sdf':
        mol = load_all_as_one_rdmol(input_path)
        obconv.SetInFormat('sdf')
    else:
        raise ValueError("Only PDB and SDF files are supported!")

    ob_suc = obconv.ReadFile(obmol, input_path)
    if not mol or not mol.GetNumConformers() or not ob_suc:
        raise ValueError("Read file failed!")

    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coordinates = conf.GetPositions()  # Nx3 numpy array

    # 初始化返回值和标志位
    res_names = []
    res_ids = []
    box_tensor = [0.0] * 9
    has_box = False
    has_res_info = False

    # ==================== 格式 A: PDB 处理分支 ====================
    if ext == 'pdb':
        # 分支 A-1: 提取 CRYST1 盒子
        with open(input_path, 'r') as f:
            for line in f:
                if line.startswith("CRYST1"):
                    try:
                        a = float(line[6:15])
                        b = float(line[15:24])
                        c = float(line[24:33])
                        # 构造简单的正交张量
                        box_tensor = [a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c]
                        has_box = True
                    except ValueError:
                        pass
                    break

        # 分支 A-2: 提取并还原（Unwrap）残基编号
        # 机制：由于 PDB 只有 4 位宽(%4d)，数字会在 9999 之后回绕到 0 或 1。
        # 我们通过监控突变，在内存中恢复真实的百万级大编号。
        last_file_rid = -1
        wrap_counter = 0

        # 预先检查第一个原子是否有 MonomerInfo
        first_atom_info = mol.GetAtomWithIdx(0).GetMonomerInfo()
        if first_atom_info:
            has_res_info = True
            for atom in mol.GetAtoms():
                info = atom.GetMonomerInfo()
                r_name = info.GetResidueName().strip()
                current_file_rid = info.GetResidueNumber()

                # 检测到回绕（例如从 9999 突变到 0 或 1）
                if last_file_rid != -1 and current_file_rid < last_file_rid:
                    wrap_counter += 1

                # 恢复真实的大编号 (假设 PDB 每 10000 一个循环)
                true_rid = current_file_rid + (wrap_counter * 10000)

                res_names.append(r_name if r_name else "RES")
                res_ids.append(true_rid)
                last_file_rid = current_file_rid

    # ==================== 格式 B: SDF 处理分支 ====================
    elif ext == 'sdf':
        # 分支 B-1: 从自定义属性标签中恢复大体系信息
        if mol.HasProp("BOX_TENSOR") and mol.HasProp("RES_NAMES") and mol.HasProp("RES_NUMS"):
            try:
                box_tensor = [float(x) for x in mol.GetProp("BOX_TENSOR").split()]
                res_names = mol.GetProp("RES_NAMES").split()
                res_ids = [int(x) for x in mol.GetProp("RES_NUMS").split()]

                if len(box_tensor) == 9 and len(res_names) == num_atoms and len(res_ids) == num_atoms:
                    has_box = True
                    has_res_info = True
            except ValueError:
                pass

    # ==================== 统一兜底机制 (Fallback) ====================
    # 如果没有读到残基信息
    if not has_res_info:
        logger.warn("No residue info detected, all residues will be named as 'UNL'，with id=1")
        res_names = ["UNL"] * num_atoms
        res_ids = [1] * num_atoms

    # 如果没有读到盒子信息
    if not has_box:
        logger.warn("No box info detected, the box is set to be +5A of max - min")
        max_coords = np.max(coordinates, axis=0)
        min_coords = np.min(coordinates, axis=0)
        # 计算跨度并加上 5.0 埃的真空层
        dx, dy, dz = max_coords - min_coords + 5.0
        box_tensor = [dx, 0.0, 0.0, 0.0, dy, 0.0, 0.0, 0.0, dz]

    return obmol, mol, coordinates, res_names, res_ids, box_tensor
