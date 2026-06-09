from openbabel import openbabel as ob
from typing import Literal

from rdkit import Chem

from misc.logger import logger
from opls.opls import opls_setup


class FF(object):
    def __init__(self, name: Literal['opls']):
        # current opls only
        self.name = name
        self.params = None
        self.rdmol = None
        self.obmol = None
        self.success = False
        self._missing = None

    def setup(self, rdmol: Chem.Mol, obmol: ob.OBMol = None, **kwargs):
        self.rdmol = rdmol
        if obmol is None:
            obmol = ob.OBMol()
            if not rdmol.HasConformer():
                logger.warn("The rdmol has no conformer, "
                            "if the molecule is large, it will be very slow or may be failed.")
            sdf_block = Chem.MolToMolBlock(rdmol, forceV3000=True)
            ob_conv = ob.OBConversion()
            ob_conv.SetInFormat("sdf")
            ob_mol = ob.OBMol()
            success = ob_conv.ReadString(ob_mol, sdf_block)
            self.obmol = ob_mol
            if not success:
                logger.warn("rdmol to obmol failed!")
                self.obmol = None

        if self.name == 'opls':
            self.params, self._missing, self.success = opls_setup(rdmol, obmol, **kwargs)

    # TODO: add MD modules
    def energy(self):
        pass

    def forces(self):
        pass

    def hessian(self):
        pass

    def optimize(self, runs: int = 1000):
        pass
