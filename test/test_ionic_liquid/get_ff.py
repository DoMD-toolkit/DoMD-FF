import sys

from misc.pipeline import run_itp_mode, run_top_mode
from misc.parser import molecule_reader
from rdkit import Chem
from rdkit.Chem import AllChem
from misc.logger import logger
logger.setLevel('WARNING')


rdmol = Chem.MolFromSmiles('O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F')
num_atoms = rdmol.GetNumAtoms()
res_names = ["T"] * num_atoms
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
run_itp_mode(rdmols, test_output_dir, molecule_name=['TFSI'])
run_top_mode(rdmol, test_output_dir, base_name="top_system", obmol=obmol)
