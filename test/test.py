
from ForceField import FF



if __name__ == '__main__':
    from rdkit import Chem
    from openbabel import openbabel as ob
    import logging
    from misc.logger import logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    #rdmol = Chem.MolFromPDBFile('test_data/test_system.sdf', removeHs=False)
    suppl = Chem.SDMolSupplier('test_data/test_system.sdf', removeHs=False)
    rdmol = None
    for mol in suppl:
        if mol is None: continue
        if rdmol is None:
            rdmol = mol
        else:
            rdmol = Chem.CombineMols(rdmol, mol)
    obmol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat('sdf')
    conv.ReadFile(obmol, 'test_data/test_system.sdf')
    forcefield = FF('opls')
    forcefield.setup(rdmol, obmol, useGMX=True, useBOSS=True)

