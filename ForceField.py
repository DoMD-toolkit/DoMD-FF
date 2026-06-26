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
        self.charges = {}

    def setup(self, rdmol: Chem.Mol, obmol: ob.OBMol = None, charge_factor: float = 1, **kwargs):
        self.rdmol = rdmol
        self.obmol = obmol
        formal_charge = Chem.GetFormalCharge(rdmol)
        logger.info(f"Formal charge of molecule {rdmol}: {formal_charge:.4f}")
        charge_drift = atom_count = 0

        if self.name == 'opls':
            self.params, self._missing, self.success = opls_setup(rdmol, obmol, **kwargs)
            if self.success:
                for idx in self.params[0]:
                    charge_drift += self.params[0][idx].charge
                    atom_count += 1
                logger.info(f"OPLS total charge for {rdmol}: {charge_drift:.4f}")
                charge_drift /= atom_count
                for idx in self.params[0]:
                    self.charges[idx] = (self.params[0][
                                             idx].charge - charge_drift + formal_charge / atom_count) * charge_factor
                logger.info(f"OPLS reset total charge to formal charge {formal_charge * charge_factor:.4f}.")

    # TODO: add MD modules
    def energy(self):
        pass

    def forces(self):
        pass

    def hessian(self):
        pass

    def optimize(self, runs: int = 1000):
        pass
