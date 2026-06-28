from ForceField import FF

if __name__ == '__main__':
    import logging
    from misc.logger import logger
    from misc.parser import molecule_reader

    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    # obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_data/test_system.pdb')
    obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_hybrid_system/test_spe_system.sdf')
    # obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_data/sbr_optimized.pdb')
    forcefield = FF('opls')
    forcefield.setup(rdmol, obmol, useGMX=False, useBOSS=True, overwrite=False, useML=True)
    params_atom, params_bonded, params_improper = forcefield.params
