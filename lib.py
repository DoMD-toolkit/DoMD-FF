from itertools import combinations

from rdkit import Chem


def count_bonded(bonded):
    m_b = m_a = m_d = 0
    for m in bonded:
        if len(m) == 2:
            m_b += 1
        if len(m) == 3:
            m_a += 1
        if len(m) == 4:
            m_d += 1
    return m_b, m_a, m_d


def get_opls_bonded_idx(rdmol: Chem.Mol):
    bond_idx, angle_idx, dihedral_idx, improper_idx = set(), set(), set(), set()
    for bond in rdmol.GetBonds():
        bond_idx.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_i, atom_j = rdmol.GetAtomWithIdx(bi), rdmol.GetAtomWithIdx(bj)
        for atom_k in atom_i.GetNeighbors():
            bk = atom_k.GetIdx()
            if bk == bj:
                continue
            for atom_l in atom_j.GetNeighbors():
                bl = atom_l.GetIdx()
                if bl in (bi, bj, bk):
                    continue
                tpl = (bk, bi, bj, bl)
                if tpl not in bond_idx:
                    dihedral_idx.add(tpl)
    for atom in rdmol.GetAtoms():
        j = atom.GetIdx()
        nbrs = [_.GetIdx() for _ in atom.GetNeighbors()]
        for i, k in combinations(nbrs, 2):
            tpl = (i, j, k)
            if tpl not in angle_idx:
                angle_idx.add(tpl)
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        if len(atom.GetNeighbors()) == 3 and atom.GetHybridization().name == 'SP2':
            neighbors = list(atom.GetNeighbors())
            # the center atom is always at j.
            i, j, k, l = neighbors[0].GetIdx(), idx, neighbors[1].GetIdx(), neighbors[2].GetIdx()
            improper_idx.add((i, j, k, l))
    return bond_idx, angle_idx, dihedral_idx, improper_idx
