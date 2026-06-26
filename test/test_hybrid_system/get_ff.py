from rdkit import Chem

from misc.logger import logger
from misc.parser import molecule_reader
from misc.pipeline import run_itp_mode, run_top_mode

logger.setLevel('WARNING')

test_output_dir = 'output'
test_file = 'test_spe_system.sdf'

rdmols = Chem.SDMolSupplier(test_file, removeHs=False)
# for mol in rdmols:
#    if mol is not None:
#        Chem.SanitizeMol(mol)
obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader(test_file)

# ---- Test Case 1: itp mode ----
run_itp_mode(rdmols, test_output_dir)
run_top_mode(rdmol, test_output_dir, base_name="top_system", obmol=obmol)
