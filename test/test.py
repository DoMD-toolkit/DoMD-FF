import sys
sys.path.append('E:\\downloads\\article\\high_throughput_system\\software\\DoMDv1.0.2\\DoMD-FF')

from ForceField import FF



if __name__ == '__main__':
    from rdkit import Chem
    from openbabel import openbabel as ob
    import logging
    from misc.logger import logger
    from misc.parser import molecule_reader
    from misc.io.gmx import write_gro_file, write_top_file, write_itp_file
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    #obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_data/test_system.pdb')
    obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_data/split_mols_fixed.sdf')
    #obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_data/sbr_optimized.pdb')
    forcefield = FF('opls')
    forcefield.setup(rdmol, obmol, useGMX=True, useBOSS=True, overwrite=False, useML=True)
    params_atom, params_bonded, params_improper =  forcefield.params
    for k in params_improper:
        print(f"{k}: {params_improper[k]}")
    #write_gro_file('test_data/test_system.gro', coordinates, res_names, res_ids, box_tensor)
    #write_top_file('test_data/test_system.top', params_atom, params_bonded, params_improper, res_names, res_ids)
    #write_itp_file('test_data/test_system.itp', params_atom, params_bonded, params_improper, res_names, res_ids)
