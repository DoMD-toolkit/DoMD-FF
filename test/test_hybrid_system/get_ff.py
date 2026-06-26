import sys
sys.path.append('E:\\downloads\\article\\high_throughput_system\\software\\DoMDv1.0.2\\DoMD-FF')

from misc.pipeline import run_itp_mode, run_top_mode
from misc.parser import molecule_reader
from rdkit import Chem
from misc.logger import logger
logger.setLevel('WARNING')

test_output_dir = 'E:\\downloads\\article\\high_throughput_system\\software\\DoMDv1.0.2\\DoMD-FF\\test\\test_hybrid_system\\output'
test_file = 'E:\\downloads\\article\\high_throughput_system\\software\\DoMDv1.0.2\\DoMD-FF\\test\\test_hybrid_system\\reconstructed_system.sdf'

rdmols = Chem.SDMolSupplier(test_file, removeHs=False)
#for mol in rdmols:
#    if mol is not None:
#        Chem.SanitizeMol(mol)
obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader(test_file)

# ---- Test Case 1: itp mode ----
run_itp_mode(rdmols, test_output_dir, base_name="itp_component")
run_top_mode(rdmol, test_output_dir, base_name="top_system", obmol=obmol)