import math
from typing import Any, Union
import tqdm
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import GetUFFBondStretchParams, GetUFFAngleBendParams, GetUFFTorsionParams
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from torch_geometric.data import Data

en = {
    "H": 2.300, "He": 4.160,"Li": 0.912, "Be": 1.576,
    "B": 2.051, "C": 2.544,"N": 3.066, "O": 3.610,
    "F": 4.193, "Ne": 4.787,"Na": 0.869, "Mg": 1.293,
    "Al": 1.613, "Si": 1.916,"P": 2.253, "S": 2.589,
    "Cl": 2.869, "Ar": 3.242,"K": 0.734, "Ca": 1.034,
    "Sc": 1.19, "Ti": 1.38,"V": 1.53, "Cr": 1.65,
    "Mn": 1.75, "Fe": 1.80,"Co": 1.84, "Ni": 1.88,
    "Cu": 1.85, "Zn": 1.588,"Ga": 1.756, "Ge": 1.994,
    "As": 2.211, "Se": 2.424,"Br": 2.685, "Kr": 2.966,
    "Rb": 0.706, "Sr": 0.963,"Y": 1.12, "Zr": 1.32,
    "Nb": 1.41, "Mo": 1.47,"Tc": 1.51, "Ru": 1.54,
    "Rh": 1.56, "Pd": 1.58,"Ag": 1.87, "Cd": 1.521,
    "In": 1.656, "Sn": 1.824,"Sb": 1.984, "Te": 2.158,
    "I": 2.359, "Xe": 2.582,"Cs": 0.659, "Ba": 0.881,
    "Lu": 1.09, "Hf": 1.16,"Ta": 1.34, "W": 1.47,
    "Re": 1.60, "Os": 1.65,"Ir": 1.68, "Pt": 1.72,
    "Au": 1.92, "Hg": 1.765,"Tl": 1.789, "Pb": 1.854,
    "Bi": 2.01, "Po": 2.19,"At": 2.39, "Rn": 2.60,
    "Fr": 0.67, "Ra": 0.89
}

def get_covalent_radius(atom):
    atomic_num = atom.GetAtomicNum()

    periodic_table = Chem.GetPeriodicTable()

    try:
        return periodic_table.GetRcovalent(atomic_num)
    except Exception:
        return 0.80

def envien(atom: Chem.Atom, rdmol: Union[Chem.Mol, Chem.RWMol]) -> float:
    bs = []
    nei_en = []
    nei_bias = []
    for nei in atom.GetNeighbors():
        nei_en.append(en[nei.GetSymbol()])
        bs.append(rdmol.GetBondBetweenAtoms(atom.GetIdx(), nei.GetIdx()).GetBondTypeAsDouble())
        bias = 0
        for nnei in nei.GetNeighbors():
            if nnei.GetIdx() == atom.GetIdx():
                continue
            bias += 0.1 * en[nnei.GetSymbol()]
        nei_en[-1] += bias
    bs = np.array(bs)
    nei_en = np.array(nei_en)
    z = bs.sum()
    if z == 0:
        return 0
    bs = bs / z

    return (bs * nei_en).sum()


def getneimasssum(atom: Chem.Atom) -> float:
    s = 0
    for nei in atom.GetNeighbors():
        s += nei.GetMass()
    return s

def g_from_networkx(g):
    edge_index = []
    bo = []
    bidx = []
    for i, j, d in g.edges(data=True):
        edge_index.extend([(i, j), (j, i)]) 
        
        bo_val = d['bo']
        bidx_val = d['bidx']
        
        bo.extend([bo_val, bo_val])
        bidx.extend([bidx_val, bidx_val])
        
    x_f = []
    orig_idx = []
    for n, d in sorted(g.nodes(data=True)):
        x_f.append(d['x_f'])
        orig_idx.append(d['orig_idx'])
    num_nodes = len(g.nodes)
    return Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long).T,
        x_f=torch.tensor(np.array(x_f), dtype=torch.float),
        orig_idx=torch.tensor(orig_idx, dtype=torch.long),
        bo=torch.tensor(np.array(bo), dtype=torch.float),
        bidx=torch.tensor(bidx, dtype=torch.long),
        num_nodes=num_nodes
    )

def g_from_networkx(g):
    edge_index = []
    bo = []
    bidx = []
    for i,j in tqdm.tqdm(g.edges,total=g.number_of_edges(),desc='g from nx 1',disable=True):
        edge_index.append((i,j))
        bo.append(g.edges[i,j]['bo'])
        bidx.append(g.edges[i,j]['bidx'])
        edge_index.append((j,i))
        bo.append(g.edges[i,j]['bo'])
        bidx.append(g.edges[i,j]['bidx'])
    x_f = []
    x_f_q = []
    orig_idx = []
    for n in tqdm.tqdm(sorted(list(g.nodes)),total=g.number_of_nodes(), desc='g from nx 2',disable=True):
        x_f.append(g.nodes[n]['x_f'])
        x_f_q.append(g.nodes[n]['x_f_q'])
        orig_idx.append(g.nodes[n]['orig_idx'])
    num_nodes=len(g.nodes)
    edge_index = torch.tensor(np.array(edge_index).T)
    x_f = torch.tensor(np.array(x_f))
    x_f_q = torch.tensor(np.array(x_f_q))
    orig_idx = torch.tensor(np.array(orig_idx))
    bo = torch.tensor(np.array(bo))
    bidx = torch.tensor(np.array(bidx))
    return Data(edge_index=edge_index,x_f=x_f,x_f_q=x_f_q,orig_idx=orig_idx,bo=bo,bidx=bidx,num_nodes=num_nodes)

def bg_from_networkx(g):
    edge_index = []
    ao = []
    aidx = []
    idx = []
    for i,j in tqdm.tqdm(g.edges,total=g.number_of_edges(),desc='bg from nx 1',disable=True):
        edge_index.append((i,j))
        ao.append(g.edges[i,j]['ao'])
        aidx.append(g.edges[i,j]['aidx'])
        idx.append(g.edges[i,j]['idx'])
        edge_index.append((j,i))
        ao.append(g.edges[i,j]['ao'])
        aidx.append(g.edges[i,j]['aidx'])
        idx.append(g.edges[i,j]['idx'])
    b_f = []
    bead_idx = []
    for n in tqdm.tqdm(sorted(list(g.nodes)),total=g.number_of_nodes(), desc='bg from nx 2',disable=True):
        b_f.append(g.nodes[n]['b_f'])
        bead_idx.append(g.nodes[n]['bead_idx'])
    num_nodes=len(g.nodes)
    edge_index = torch.tensor(np.array(edge_index).T)
    b_f = torch.tensor(np.array(b_f))
    bead_idx = torch.tensor(np.array(bead_idx))
    ao = torch.tensor(np.array(ao))
    aidx = torch.tensor(np.array(aidx))
    idx = torch.tensor(np.array(idx))
    return Data(edge_index=edge_index,b_f=b_f,bead_idx=bead_idx,ao=ao,aidx=aidx,idx=idx,num_nodes=num_nodes)

def ag_from_networkx(g):
    edge_index = []
    do = []
    didx = []
    idx = []
    for i,j in tqdm.tqdm(g.edges,total=g.number_of_edges(),desc='ag from nx 1',disable=True):
        edge_index.append((i,j))
        do.append(g.edges[i,j]['do'])
        didx.append(g.edges[i,j]['didx'])
        idx.append(g.edges[i,j]['idx'])

        edge_index.append((j,i))
        do.append(g.edges[i,j]['do'])
        didx.append(g.edges[i,j]['didx'])
        idx.append(g.edges[i,j]['idx'])
    a_f = []
    bead_idx = []
    for n in tqdm.tqdm(sorted(list(g.nodes)),total=g.number_of_nodes(), desc='ag from nx 2',disable=True):
        a_f.append(g.nodes[n]['a_f'])
        bead_idx.append(g.nodes[n]['bead_idx'])
    num_nodes=len(g.nodes)
    edge_index = torch.from_numpy(np.array(edge_index).T)
    #print('torch tensor edge index done')
    a_f = torch.from_numpy(np.array(a_f))
    #print('torch tensor a_f done')
    bead_idx = torch.from_numpy(np.array(bead_idx))
    #print('torch tensor bead idx done')
    do = torch.from_numpy(np.array(do))
    #print('torch tensor do done')
    didx = torch.from_numpy(np.array(didx))
    #print('torch tensor didx done')
    idx = torch.from_numpy(np.array(idx))
    #print('torch tensor idx done')
    return Data(edge_index=edge_index,a_f=a_f,bead_idx=bead_idx,do=do,didx=didx,idx=idx,num_nodes=num_nodes)

def _get_atom_features(idx: int, mol: Chem.Mol) -> np.ndarray:
    r"""
    Extracts a 10-dimensional feature vector for a single atom.

    Features include:
    [Gasteiger partial charge, atomic number, aromaticity, ring status,
    formal charge, hybridization, explicit valence, neighbor mass sum,
    electronegativity, weighted environment electronegativity].
    """
    atom = mol.GetAtomWithIdx(idx)
    gei = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
    if math.isnan(gei): gei = 0.0

    eni = en.get(atom.GetSymbol(), 0.0)
    neni = float(envien(atom, mol))
    hi = atom.GetHybridization()

    return np.array([
        100 * gei,
        atom.GetAtomicNum(),
        int(atom.GetIsAromatic()) * 10,
        int(atom.IsInRing()) * 10,
        atom.GetFormalCharge() * 5,
        2 * hi.real + hi.imag,
        atom.GetValence(which=Chem.rdchem.ValenceType.EXPLICIT) * 5,
        getneimasssum(atom),
        eni * 5,
        neni * 5
    ], dtype=float)

def _get_atom_features_argumented(idx: int, mol: Chem.Mol) -> np.ndarray:
    r"""
    Extracts a 10-dimensional feature vector for a single atom.

    Features include:
    [Gasteiger partial charge, atomic number, aromaticity, ring status,
    formal charge, hybridization, explicit valence, neighbor mass sum,
    electronegativity, weighted environment electronegativity].
    """
    atom = mol.GetAtomWithIdx(idx)
    gei = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
    if math.isnan(gei): gei = 0.0

    eni = en.get(atom.GetSymbol(), 0.0)
    neni = float(envien(atom, mol))
    hi = atom.GetHybridization()

    return np.array([
        100 * gei,
        atom.GetAtomicNum(),
        int(atom.GetIsAromatic()) * 10,
        int(atom.IsInRing()) * 10,
        atom.GetFormalCharge() * 5,
        2 * hi.real + hi.imag,
        atom.GetValence(which=Chem.rdchem.ValenceType.EXPLICIT) * 5,
        getneimasssum(atom),
        eni / 4 * 10,
        neni * 5,
        get_covalent_radius(atom) * 10
    ], dtype=float)

def _get_bond_features(i:int, j:int, mol: Chem.Mol) -> tuple:
    r"""
    Extracts chemical bond features and UFF (Universal Force Field) parameters.

    Returns a numpy array representing the bond order and its equilibrium
    bond length predicted by UFF.
    """
    #i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bond = mol.GetBondBetweenAtoms(i,j)
    cc = GetUFFBondStretchParams(mol, i, j)
    k, bl = cc if cc is not None else (10000.0, 0.25)

    bo_feat = np.array([bond.GetBondTypeAsDouble(), bl], dtype=float)
    return bo_feat

def _build_atom_graph(mol: Chem.Mol) -> nx.Graph:
    r"""
    Builds the base atom graph where nodes represent atoms and edges represent chemical bonds.
    Populates node features (x_f) and edge features (bo) simultaneously.
    """
    g = nx.Graph()

    # 1. Populate nodes
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        g.add_node(idx, x_f=_get_atom_features(idx, mol), x_f_q=_get_atom_features_argumented(idx, mol), orig_idx=idx)

    # 2. Populate edges
    for bidx, bond in enumerate(mol.GetBonds()):
        if not bond.GetBondTypeAsDouble():
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bo_feat = _get_bond_features(i, j, mol)
        g.add_edge(i, j, bo=bo_feat, bidx=bidx)
    return g

def _build_line_graphs(g: nx.Graph, mol: Chem.Mol) -> tuple[nx.Graph, nx.Graph]:
    r"""
    Constructs the Bond Graph and Angle Graph via line graph transformation.
    
    - Bond Graph: Nodes are bonds from the atom graph; edges represent bond angles.
    - Angle Graph: Nodes are bond angles; edges represent dihedral (torsion) angles.
    
    Automatically extracts and assigns force constants and equilibrium angles/dihedrals 
    based on the UFF force field.
    """
    bond_g = nx.Graph()
    ang_g = nx.Graph()
    
    ang_set = set()
    aidx = 0
    
    # --- 1. Build Bond Graph (Nodes = Bonds, Edges = Angles) ---
    for i, j, edge_data in g.edges(data=True):
        bidx = edge_data['bidx']
        xf_i, xf_j = g.nodes[i]['x_f'], g.nodes[j]['x_f']
        
        # Initialize bond node
        if not bond_g.has_node(bidx):
            bond_g.add_node(bidx, b_f=np.concatenate((xf_i, xf_j)), bead_idx=(i, j))
            
        # Search for adjacent bonds to form angles
        for n in g.neighbors(i):
            if n == j: continue
            nidx = g.edges[i, n]['bidx']
            xf_n = g.nodes[n]['x_f']
            
            if not bond_g.has_node(nidx):
                bond_g.add_node(nidx, b_f=np.concatenate((xf_i, xf_n)), bead_idx=(i, n))
                
            # Extract UFF angle parameters
            cc = GetUFFAngleBendParams(mol, n, i, j)
            k0, an = cc if cc is not None else (1.5, 109.5)
            
            # Add angle edge (avoiding duplicates)
            if (n, i, j) not in ang_set and (j, i, n) not in ang_set:
                ang_set.add((n, i, j))
                bond_g.add_edge(bidx, nidx, ao=np.array([round(k0, 3), an], dtype=float), aidx=aidx, idx=(n, i, j))
                # Simultaneously initialize angle graph node
                ang_g.add_node(aidx, a_f=np.concatenate((xf_n, xf_i, xf_j)), bead_idx=(n, i, j))
                aidx += 1
        for n in g.neighbors(j):
            if n == i: continue
            nidx = g.edges[j, n]['bidx']
            xf_n = g.nodes[n]['x_f']

            if not bond_g.has_node(nidx):
                bond_g.add_node(nidx, b_f=np.concatenate((xf_j, xf_n)), bead_idx=(j, n))

            # Extract UFF angle parameters
            cc = GetUFFAngleBendParams(mol, n, j, i)
            k0, an = cc if cc is not None else (1.5, 109.5)

            # Add angle edge (avoiding duplicates)
            if (n, j, i) not in ang_set and (i, j, n) not in ang_set:
                ang_set.add((n, j, i))
                bond_g.add_edge(bidx, nidx, ao=np.array([round(k0, 3), an], dtype=float), aidx=aidx, idx=(n, j, i))
                # Simultaneously initialize angle graph node
                ang_g.add_node(aidx, a_f=np.concatenate((xf_n, xf_j, xf_i)), bead_idx=(n, j, i))
                aidx += 1

    # --- 2. Build Angle Graph (Nodes = Angles, Edges = Dihedrals) ---
    # Hash map for fast O(1) angle index lookup
    beadidx_to_aidx = {data['bead_idx']: node for node, data in ang_g.nodes(data=True)}
    beadidx_to_aidx.update({(k, j, i): node for node, (i, j, k) in [ (n, d['bead_idx']) for n, d in ang_g.nodes(data=True)]})
    
    didx = 0
    for i, j in g.edges():
        for ni in g.neighbors(i):
            if ni == j: continue
            for nj in g.neighbors(j):
                if nj == i: continue
                
                # Fetch indices of the two adjacent angles forming the dihedral
                nij_idx = beadidx_to_aidx[(ni, i, j)]
                nji_idx = beadidx_to_aidx[(nj, j, i)]
                
                # Extract UFF dihedral parameters
                k0 = GetUFFTorsionParams(mol, ni, i, j, nj)
                k0 = k0 if k0 is not None else 1.5
                
                ang_g.add_edge(nij_idx, nji_idx, do=round(k0, 3), didx=didx, idx=(ni, i, j, nj))
                didx += 1
                
    return bond_g, ang_g

def mol2torch_graph(molecule: Union[Chem.Mol, Chem.RWMol], debug: bool = False) -> tuple[Data, Data, Data]:
    r"""
    Converts an RDKit molecule into a three-tier hierarchical graph structure
    (Atom Graph, Bond Graph, and Angle Graph) required for Machine Learning Force Fields (MLFF).

    Args:
        molecule: The input RDKit molecule object.
        debug: If True, runs strict sanity checks to prevent topological loss.

    Returns:
        A tuple of three PyTorch Geometric Data objects corresponding to the
        atom, bond, and angle graphs respectively.
    """
    ComputeGasteigerCharges(molecule, nIter=120)

    # 1. Construct the base atom graph
    g = _build_atom_graph(molecule)

    # 2. Perform line-graph ascending transformations
    bond_g, ang_g = _build_line_graphs(g, molecule)

    # 3. Optional debug checking
    if debug:
        _sanity_check(g, ang_g, molecule)

    # 4. Export to PyG Tensor format
    return g_from_networkx(g), bg_from_networkx(bond_g), ag_from_networkx(ang_g)

def _sanity_check(g: nx.Graph, ang_g: nx.Graph, molecule: Chem.Mol):
    r"""
    Strict sanity check to ensure no bond angles or dihedral angles
    were omitted or structurally orphaned during the line graph transformations.
    """
    import tqdm

    error_flag = False
    idx_di = set()

    for e in ang_g.edges:
        ni, i, j, nj = ang_g.edges[e]['idx']
        idx_di.add((ni, i, j, nj))
        idx_di.add((nj, j, i, ni))

    for b in tqdm.tqdm(molecule.GetBonds(),total=molecule.GetNumBonds(),desc='check bond',disable=True):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # for ni in g.neighbors(i):
        for nbri in molecule.GetAtomWithIdx(i).GetNeighbors():
            ni = nbri.GetIdx()
            if ni == j:
                continue
            # for nj in g.neighbors(j):
            for nbrj in molecule.GetAtomWithIdx(j).GetNeighbors():
                nj = nbrj.GetIdx()
                if nj == i or nj == ni:
                    continue
                if (ni, i, j, nj) not in idx_di:
                    print(ni, i, j, nj)
                    error_flag = True
                    print('dihedral error')

    idx_an = set()
    for n in ang_g.nodes:
        ni, i, nj = ang_g.nodes[n]['bead_idx']
        idx_an.add((ni, i, nj))
        idx_an.add((nj, i, ni))

    for i in g.nodes:
        if g.degree(i) >= 2:
            neis = g.neighbors(i)
            for ni in neis:
                for nj in neis:
                    if ni == nj:
                        continue
                    if (ni, i, nj) not in idx_an:
                        # print(ni, i, nj)
                        error_flag = True
                        print('angle error')

    if error_flag:
        raise Exception('ml graph creation error')




