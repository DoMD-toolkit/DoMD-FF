import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
from typing import List, Union
import tqdm
import networkx as nx

import sys
sys.path.append('/Users/zhangxuze/DoMD_v1.0.2/DoMD-FF/')
from misc.logger import logger



bondorder_to_type = {
        0: Chem.rdchem.BondType.UNSPECIFIED,
        1: Chem.rdchem.BondType.SINGLE,
        1.5:Chem.rdchem.BondType.AROMATIC,
        2:Chem.rdchem.BondType.DOUBLE,
        3:Chem.rdchem.BondType.TRIPLE
}

def mols_to_nxgraphs(molecules: List[Union[Chem.Mol, Chem.RWMol]]):
    """Converts a list of RDKit molecules into NetworkX graphs.



        Preserves atomic properties (element, mass, charge, aromaticity) and custom
        residue identifiers (`res_id`, `global_res_id`, `res_name`) as node attributes.
        Bond orders are stored as edge attributes.

        Args:
            molecules (List[Union[Chem.Mol, Chem.RWMol]]): List of RDKit molecule objects.

        Returns:
            List[nx.Graph]: A list of NetworkX graphs representing the molecules.
    """
    mols_meta = []
    for mol in tqdm.tqdm(molecules,total=len(molecules),desc='converting Chem.Mol to nx.Graph',disable=True):
        mol_meta = nx.Graph()
        nodes = set()
        tqdm_show = True
        #if mol.GetNumAtoms() > 5000:
        #    tqdm_show = True
        for ai in tqdm.tqdm(mol.GetAtoms(),total=mol.GetNumAtoms(),desc='adding atoms',disable=tqdm_show):
            i = ai.GetIdx()
            if i not in nodes:
                mol_meta.add_node(i, element =ai.GetSymbol(),
                                     atomic_num = ai.GetAtomicNum(),
                                     mass =ai.GetMass(),
                                     charge =ai.GetFormalCharge(),
                                     is_aromatic=ai.GetIsAromatic(),
                                     res_name =ai.GetProp('res_name'),
                                     res_id =ai.GetIntProp('res_id'),
                                     global_res_id =ai.GetIntProp('global_res_id')
                                )

        for bond in tqdm.tqdm(mol.GetBonds(),total=mol.GetNumBonds(),desc='adding bonds', disable=tqdm_show):
            ai, aj = bond.GetBeginAtom(), bond.GetEndAtom()
            i, j = ai.GetIdx(), aj.GetIdx()
            mol_meta.add_edge(i,j, bondorder=bond.GetBondTypeAsDouble())
        mols_meta.append(mol_meta)
    return mols_meta

def nxgraphs_to_mols(mols_meta: List[nx.Graph]):
    """Reconstructs RDKit molecules from NetworkX graphs.

        Reverse operation of `mols_to_nxgraphs`. Rebuilds the molecule structure and
        restores atomic properties and bond types based on the graph attributes.

        Args:
            mols_meta (List[nx.Graph]): List of NetworkX graphs with molecular metadata.

        Returns:
            List[Chem.Mol]: A list of sanitized RDKit molecules.
    """
    molecules = []
    for g in tqdm.tqdm(mols_meta,total=len(mols_meta),desc='converting nx.Graph to Chem.Mol',disable=True):
        mol = Chem.RWMol()
        nodes = g.nodes
        n_to_aid = {}
        for n in g.nodes:
            atom = Chem.Atom(nodes[n]['atomic_num'])
            atom.SetIsAromatic(nodes[n]['is_aromatic'])
            atom.SetFormalCharge(nodes[n]['charge'])
            atom.SetIntProp('res_id',nodes[n]['res_id'])
            atom.SetIntProp('global_res_id',nodes[n]['global_res_id'])
            atom.SetProp('res_name',nodes[n]['res_name'])
            aid = mol.AddAtom(atom)
            n_to_aid[n] = aid
        edges = g.edges
        for i,j in g.edges:
            mol.AddBond(
                    n_to_aid[i],
                    n_to_aid[j],
                    bondorder_to_type[edges[(i,j)]['bondorder']]
                    )
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
        molecules.append(mol)
    return molecules


def write_mols_to_sdf(mols, output_path, force_v3000=True):
    """
    Writes a list of RDKit molecule objects into a single multi-molecule SDF file.
    Automatically preserves all atom properties and custom string metadata.

    Parameters:
        mols (list): List of RDKit Romol objects.
        output_path (str): Target path for the output .sdf file.
        force_v3000 (bool): If True, enforces V3000 format compliance (recommended for large systems).
    """
    # Initialize the SDWriter handler
    writer = Chem.SDWriter(output_path)

    if force_v3000:
        writer.SetForceV3000(True)

    for idx, mol in tqdm.tqdm(enumerate(mols),total=len(mols),desc='writing molecules to SDF',disable=False):
        if mol is None:
            continue
        # SDWriter automatically reads and writes all tags set via mol.SetProp()
        writer.write(mol)

    # Crucial: Always close the stream to flush buffer and finalize the file structure
    writer.close()

confs, mol_graphs, box = pickle.load(open("meta_aa_top.pkl", "rb"))

rdmols = nxgraphs_to_mols(mol_graphs)
#atom_numbers = [mol.GetNumAtoms() for mol in rdmols]
#atom_numbers = sorted(atom_numbers)
#print(atom_numbers[-10:])
for i, mol in enumerate(rdmols):
    RES_NAMES = []
    RES_NUMS = []
    for atom in mol.GetAtoms():
        RES_NAMES.append(atom.GetProp("res_name"))
        RES_NUMS.append(str(atom.GetIntProp("res_id")))
    mol.SetProp("RES_NAMES", " ".join(RES_NAMES))
    mol.SetProp("RES_NUMS", " ".join(RES_NUMS))
    box_tensor = f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} 0.000000 0.000000 0.000000"
    mol.SetProp("BOX_TENSOR", box_tensor)
    positions = confs[i]
    num_atoms = mol.GetNumAtoms()
    mol.RemoveAllConformers()
    rd_conf = Chem.Conformer(num_atoms)
    for atom_idx in range(num_atoms):
        xyz = positions[atom_idx]
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        rd_conf.SetAtomPosition(atom_idx, Point3D(x, y, z))
    mol.AddConformer(rd_conf, assignId=True)
#rdmols_small = []
#for mol in rdmols:
#    if mol.GetNumAtoms() > 10000:
#        continue
#    rdmols_small.append(mol)
write_mols_to_sdf(rdmols, "reconstructed_system.sdf", force_v3000=True)

