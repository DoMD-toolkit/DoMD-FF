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
        self._meta = None

    def setup(self, rdmol: Chem.Mol, obmol: ob.OBMol = None, charge_factor: float = 1, **kwargs):
        self.rdmol = rdmol
        self.obmol = obmol
        formal_charge = Chem.GetFormalCharge(rdmol)
        logger.debug(f"Formal charge of molecule {rdmol}: {formal_charge:.4f}")

        if self.name == 'opls':
            self.params, self._missing, self.success, self._meta = opls_setup(rdmol, obmol, **kwargs)
            if self.success:
                ion_indices = []
                non_ion_indices = []
                for atom in rdmol.GetAtoms():
                    idx = atom.GetIdx()
                    if atom.GetDegree() == 0:
                        ion_indices.append(idx)
                    else:
                        non_ion_indices.append(idx)
                atom_count = len(self.params[0])

                total_opls_charge = sum(float(self.params[0][idx].charge) for idx in self.params[0])
                logger.debug(f"OPLS raw total charge for {rdmol}: {total_opls_charge:.4f}")

                global_drift_per_atom = total_opls_charge / atom_count
                temp_charges = {}
                for idx in self.params[0]:
                    temp_charges[idx] = self.params[0][idx].charge - global_drift_per_atom + (
                                formal_charge / atom_count)

                need_ion_constraint = False
                for idx in ion_indices:
                    atom_formal_charge = rdmol.GetAtomWithIdx(idx).GetFormalCharge()
                    if temp_charges[idx] > atom_formal_charge:
                        need_ion_constraint = True
                        logger.debug(f"Ion constraint triggered by atom {idx} "
                                     f"(Calculated: {temp_charges[idx]:.4f} > Formal: {atom_formal_charge})")
                        break


                if not need_ion_constraint:
                    for idx in self.params[0]:
                        self.charges[idx] = temp_charges[idx] * charge_factor
                    logger.info(f"OPLS reset total charge to formal charge "
                                 f"{formal_charge * charge_factor:.4f} (Global uniform distribution).")

                else:
                    ion_total_charge = 0.0
                    for idx in ion_indices:
                        atom_fc = float(rdmol.GetAtomWithIdx(idx).GetFormalCharge())
                        self.charges[idx] = atom_fc * charge_factor
                        ion_total_charge += atom_fc

                    if non_ion_indices:
                        target_non_ion_charge = formal_charge - ion_total_charge

                        non_ion_opls_charge = sum(float(self.params[0][idx].charge) for idx in non_ion_indices)
                        non_ion_count = len(non_ion_indices)
                        non_ion_drift_per_atom = non_ion_opls_charge / non_ion_count

                        for idx in non_ion_indices:
                            self.charges[idx] = (self.params[0][idx].charge - non_ion_drift_per_atom + (
                                        target_non_ion_charge / non_ion_count)) * charge_factor
                    logger.info(f"OPLS reset total charge to formal charge {formal_charge * charge_factor:.4f} "
                                f"(Ion constrained, {len(non_ion_indices)} non-ions adjusted).")

    # TODO: add MD modules
    def energy(self):
        pass

    def forces(self):
        pass

    def hessian(self):
        pass

    def optimize(self, runs: int = 1000):
        pass
