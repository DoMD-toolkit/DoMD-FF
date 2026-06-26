import sys

from misc.pipeline import run_itp_mode, run_top_mode
from misc.parser import molecule_reader
from rdkit import Chem
from rdkit.Chem import AllChem
from misc.logger import logger
logger.setLevel('WARNING')


def normalize_mol_topology(rdmol):
    """
    通过将分子序列化为规范 SMILES 再写回，
    强行抹平由于读取自不同文件（SDF/PDB/SMILES）造成的键级、芳香性和隐式氢的表示差异。
    """
    if rdmol is None:
        return None

    # 1. 强行转换为带电荷和化学语义的规范 SMILES（这会强制 RDKit 重新对齐所有键级）
    canonical_smiles = Chem.MolToSmiles(rdmol, canonical=True)

    # 2. 重新从规范 SMILES 读回，生成一个处于标准价态底座上的新分子
    standard_mol = Chem.MolFromSmiles(canonical_smiles)

    # 3. 补回显式氢原子（如果你的参数化流程需要显式氢，必须要加，否则特征图对不上）
    standard_mol = Chem.AddHs(standard_mol)

    # 4. 如果需要，把原分子的 3D 坐标强行映射回这个标准分子（防止丢失坐标）
    if rdmol.GetNumConformers() > 0:
        standard_mol.RemoveAllConformers()
        standard_mol.AddConformer(rdmol.GetConformer(), assignId=True)

    return standard_mol

rdmol = Chem.MolFromSmiles('O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F')
rdmol = normalize_mol_topology(rdmol)
num_atoms = rdmol.GetNumAtoms()
res_names = ["UNL"] * num_atoms
res_ids = [1] * num_atoms
box_tensor = [50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
rdmol.SetProp("RES_NAMES", " ".join(res_names))
rdmol.SetProp("RES_NUMS", " ".join(map(str, res_ids)))
rdmol.SetProp("BOX_TENSOR", " ".join(map(str, box_tensor)))
AllChem.EmbedMolecule(rdmol)
AllChem.UFFOptimizeMolecule(rdmol)
obmol = None
rdmols = [rdmol]
# ---- Test Case 1: itp mode ----
test_output_dir = 'output'
run_itp_mode(rdmols, test_output_dir)
run_top_mode(rdmol, test_output_dir, base_name="top_system", obmol=obmol)
