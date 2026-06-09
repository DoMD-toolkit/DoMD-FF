
from ForceField import FF



if __name__ == '__main__':
    from rdkit import Chem
    from openbabel import openbabel as ob
    import logging
    from misc.logger import logger
    logger.setLevel(logging.INFO)

    rdmol = Chem.MolFromPDBFile('test_data/sbr_optimized.pdb', removeHs=False)
    obmol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat('pdb')
    conv.ReadFile(obmol, 'test_data/sbr_optimized.pdb')
    forcefield = FF('opls')
    forcefield.setup(rdmol, obmol, useGMX=True, useBOSS=True)

