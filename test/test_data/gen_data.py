import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_perfect_separated_test_systems(pdb_out="test_system.pdb", sdf_out="test_system.sdf"):
    mol1 = Chem.MolFromSmiles("c1ccccc1CC")
    mol1 = Chem.AddHs(mol1)
    AllChem.EmbedMolecule(mol1, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol1)

    mol2 = Chem.MolFromSmiles("c1ccccc1")
    mol2 = Chem.AddHs(mol2)
    AllChem.EmbedMolecule(mol2, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol2)


    conf2 = mol2.GetConformer()
    for i in range(mol2.GetNumAtoms()):
        pos = conf2.GetAtomPosition(i)
        conf2.SetAtomPosition(i, (pos.x + 15.0, pos.y, pos.z))


    combined_mol = Chem.CombineMols(mol1, mol2)


    benzene_query = Chem.MolFromSmarts("c1ccccc1")
    ring_matches = combined_mol.GetSubstructMatches(benzene_query)

    atom_res_map = {}
    res_id_counter = 1

    for match in ring_matches:
        for idx in match:
            if idx not in atom_res_map:
                atom_res_map[idx] = ("PH", res_id_counter)
        res_id_counter += 1


    for atom in combined_mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() > 1 and idx not in atom_res_map:
            atom_res_map[idx] = ("CC", res_id_counter)
    res_id_counter += 1


    for atom in combined_mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 1:
            neighbors = atom.GetNeighbors()
            if neighbors:
                heavy_neighbor_idx = neighbors[0].GetIdx()
                atom_res_map[idx] = atom_res_map[heavy_neighbor_idx]
            else:
                atom_res_map[idx] = ("UNL", 999)

    for i, atom in enumerate(combined_mol.GetAtoms()):
        res_name, res_id = atom_res_map[i]
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(res_name.ljust(4))
        info.SetResidueNumber(res_id)
        info.SetName(f"{atom.GetSymbol()}{i + 1}".ljust(4))
        atom.SetMonomerInfo(info)


    a, b, c = 40.0, 40.0, 40.0
    alpha, beta, gamma = 75.0, 85.0, 105.0

    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    v1 = [a, 0.0, 0.0]
    v2_x = b * np.cos(gamma_rad)
    v2_y = b * np.sin(gamma_rad)
    v2 = [v2_x, v2_y, 0.0]

    v3_x = c * np.cos(beta_rad)
    v3_y = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    v3_z = np.sqrt(c ** 2 - v3_x ** 2 - v3_y ** 2)
    v3 = [v3_x, v3_y, v3_z]

    box_tensor = v1 + v2 + v3

    cryst1_line = f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
    pdb_block = Chem.MolToPDBBlock(combined_mol, flavor=4)
    with open(pdb_out, "w") as f:
        f.write(cryst1_line + pdb_block)

    num_atoms = combined_mol.GetNumAtoms()
    res_names_list = [atom_res_map[i][0] for i in range(num_atoms)]
    res_ids_list = [str(atom_res_map[i][1]) for i in range(num_atoms)]

    combined_mol.SetProp("BOX_TENSOR", " ".join(f"{x:.6f}" for x in box_tensor))
    combined_mol.SetProp("RES_NAMES", " ".join(res_names_list))
    combined_mol.SetProp("RES_NUMS", " ".join(res_ids_list))

    # with open(sdf_out, "w") as f:
    #     f.write(Chem.MolToMolBlock(combined_mol, forceV3000=True))
    #     f.write(f">  <BOX_TENSOR>\n{combined_mol.GetProp('BOX_TENSOR')}\n\n")
    #     f.write(f">  <RES_NAMES>\n{combined_mol.GetProp('RES_NAMES')}\n\n")
    #     f.write(f">  <RES_NUMS>\n{combined_mol.GetProp('RES_NUMS')}\n\n")
    #     f.write("$$$$\n")

    writer = Chem.SDWriter(sdf_out)
    writer.SetForceV3000(True)
    combined_mol.SetProp('BOX_TENSOR', str(combined_mol.GetProp('BOX_TENSOR')))
    combined_mol.SetProp('RES_NAMES', str(combined_mol.GetProp('RES_NAMES')))
    combined_mol.SetProp('RES_NUMS', str(combined_mol.GetProp('RES_NUMS')))

    writer.write(combined_mol)
    writer.close()



if __name__ == "__main__":
    generate_perfect_separated_test_systems()