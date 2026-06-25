
from ForceField import FF



if __name__ == '__main__':
    from rdkit import Chem
    from openbabel import openbabel as ob
    import logging
    from misc.logger import logger
    from misc.parser import molecule_reader
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    # obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_data/test_system.pdb')
    obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_split_output/split_mols_fixed.sdf')
    forcefield = FF('opls')
    forcefield.setup(rdmol, obmol, useGMX=True, useBOSS=True)

