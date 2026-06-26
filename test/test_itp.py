import sys
sys.path.append('E:\\downloads\\article\\high_throughput_system\\software\\DoMDv1.0.2\\DoMD-FF')
import os
from openbabel import pybel as pb
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolHash

from ForceField import FF

import logging
from misc.logger import logger

logger.setLevel(logging.INFO)
logger.propagate = True


def align_molecule_order(mol_A, mol_B):
    # size should be limited for match
    match = mol_B.GetSubstructMatch(mol_A, useChirality=True)

    if not match:
        raise ValueError("Error!")

    if len(match) != mol_A.GetNumAtoms():
        raise ValueError("Error!")

    mol_B_reordered = Chem.RenumberAtoms(mol_B, match)

    return mol_B_reordered


def parse_and_split_system(input_sdf, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)
    raw_mols = [m for m in suppl if m is not None]
    combined_mol = raw_mols[0]
    if not raw_mols:
        raise ValueError("RDKIT parse failed!")
    for m in raw_mols[1:]:
        combined_mol = Chem.CombineMols(combined_mol, m)

    all_fragments = []
    for m in raw_mols:
        frags = Chem.GetMolFrags(m, asMols=True)
        all_fragments.extend(frags)

    global_res_names = combined_mol.GetProp('RES_NAMES').split()  # 得到 ['PH', 'PH', ..., 'PH']
    global_res_nums = combined_mol.GetProp('RES_NUMS').split()

    writer = Chem.SDWriter(f"{work_dir}/split_mols_fixed.sdf")
    writer.SetForceV3000(True)

    atom_pointer = 0

    for mol in all_fragments:
        num_atoms = mol.GetNumAtoms()

        local_names = global_res_names[atom_pointer: atom_pointer + num_atoms]
        local_nums = global_res_nums[atom_pointer: atom_pointer + num_atoms]

        mol.SetProp('RES_NAMES', " ".join(local_names))
        mol.SetProp('RES_NUMS', " ".join(local_nums))

        atom_pointer += num_atoms
        writer.write(mol)

    writer.close()


def encode_by_appearance(arr):
    mapping = {}
    result = []
    current_id = 1

    for num in arr:
        if num not in mapping:
            mapping[num] = current_id
            current_id += 1
        result.append(mapping[num])

    return result


def run_massive_system_pipeline(input_sdf, work_dir, box_tensor=None):
    os.makedirs(work_dir, exist_ok=True)

    rd_suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)
    ob_suppl = pb.readfile("sdf", input_sdf)

    mol_registry = {}
    large_counter = 0
    small_type_map = {}  # {wl_hash: type_name}
    small_counter = 0

    for rdmol, pbmol in zip(rd_suppl, ob_suppl):
        obmol = pbmol.OBMol
        assert rdmol.GetNumAtoms() == obmol.NumAtoms()
        res_names = ['UNL'] * rdmol.GetNumAtoms()
        res_ids = [1] * rdmol.GetNumAtoms()
        if rdmol.HasProp("RES_NAMES"):
            res_names = rdmol.GetProp("RES_NAMES").split()
        if rdmol.HasProp("RES_NUMS"):
            res_ids = encode_by_appearance([int(x) for x in rdmol.GetProp("RES_NUMS").split()]) # remap to [1,1,2,...

        if rdmol.GetNumAtoms() >= 1000:
            mol_registry[f'large_{large_counter}'] = {"ref_rdmol": rdmol,
                                                      "ref_obmol": obmol,
                                                      "instances": [(rdmol, obmol)],
                                                      'large_p': True,
                                                      'res_names': res_names,
                                                      'res_ids': res_ids}
            large_counter += 1
        else:
            wl_hash = rdMolHash.MolHash(rdmol, rdMolHash.HashFunction.AnonymousGraph)
            if wl_hash not in small_type_map:
                mol_registry[f'small_{small_counter}'] = {"ref_rdmol": rdmol,
                                                          "ref_obmol": obmol,
                                                          "instances": [(rdmol, obmol)],
                                                          'large_p': False,
                                                          'res_names': res_names,
                                                          'res_ids': res_ids}
                small_type_map[wl_hash] = f'small_{small_counter}'
            else:
                mol_registry[small_type_map[wl_hash]]["instances"].append((rdmol, obmol))
            small_counter += 1

    final_ordered_molecules = []
    scheme_list = []  # [(molecule_type_name, count), ...]

    for t_name, info in mol_registry.items():
        ref_mol = info["ref_rdmol"]
        instances = info["instances"]
        scheme_list.append((t_name, len(instances)))

        if info["large_p"]:
            final_ordered_molecules.extend(instances)
            continue

        remapped_instances = []
        for inst_ in instances:
            inst = inst_[0]
            match = inst.GetSubstructMatch(ref_mol)
            if match and len(match) == ref_mol.GetNumAtoms():
                reordered_inst = Chem.RenumberAtoms(inst, list(match))
                remapped_instances.append(reordered_inst)
            else:
                remapped_instances.append(inst)

        final_ordered_molecules.extend(remapped_instances)
        mol_registry[t_name]["instances"] = remapped_instances

    combined_system_mol = final_ordered_molecules[0]
    for next_mol in final_ordered_molecules[1:]:
        combined_system_mol = Chem.CombineMols(combined_system_mol, next_mol)

    global_res_names = []
    global_res_ids = []
    current_global_resid = 1

    for t_name, info in mol_registry.items():
        for _ in info["instances"]:
            n_res = len(set(info['res_ids']))
            global_res_names.extend(info['res_names'])
            global_res_ids.extend(list(np.array(info['res_ids']) + current_global_resid))
            current_global_resid += n_res

    output_gro_path = os.path.join(work_dir, "system.gro")
    conf = combined_system_mol.GetConformer()
    coords = np.array(conf.GetPositions()) - np.array(conf.GetPositions()).min()

    if rdmol.HasProp("BOX_TENSOR"):
        box_tensor = [float(x) for x in rdmol.GetProp("BOX_TENSOR").split()]
    else:
        box_tensor = [50.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 50.0]

    write_gro_file_stream(output_gro_path, combined_system_mol, coords, global_res_names, global_res_ids, box_tensor)

    all_unique_atomtypes = {}
    itp_include_lines = []

    for t_name, info in mol_registry.items():
        rd_test_mol = info['ref_rdmol']
        ob_test_mol = info['ref_obmol']

        ff = FF('opls')
        ff.setup(rd_test_mol, ob_test_mol, useGMX=True, useBOSS=True)
        p_atom, p_bonded, p_improper = ff.params

        for o_atom in p_atom.values():
            if o_atom.bond_type not in all_unique_atomtypes:
                all_unique_atomtypes[o_atom.bond_type] = o_atom

        mol_itp_name = f"{t_name}.itp"
        mol_itp_path = os.path.join(work_dir, mol_itp_name)
        write_single_molecule_itp(mol_itp_path, rd_test_mol, t_name, info['res_ids'],
                                  info['res_names'], p_atom, p_bonded, p_improper)
        itp_include_lines.append(f'#include "{mol_itp_name}"')


    atom_types_path = os.path.join(work_dir, "atom_types.itp")
    write_atom_types_itp(atom_types_path, all_unique_atomtypes)

    top_path = os.path.join(work_dir, "system.top")
    write_master_top_file(top_path, itp_include_lines, scheme_list)



def write_gro_file_stream(path, rdmol, coords, res_names, res_ids, box_tensor):
    num_atoms = len(coords)
    with open(path, 'w', buffering=65536) as f:
        f.write("Generated by Massive Topology Assembly Pipeline\n")
        f.write(f"{num_atoms:5d}\n")
        for i in range(num_atoms):
            res_id_safe = int(res_ids[i]) % 100000
            atom_id_safe = (i + 1) % 100000
            f.write(
                f"{res_id_safe:5d}{str(res_names[i])[:5]:<5}{f'{rdmol.GetAtomWithIdx(i).GetSymbol()}{i + 1}'[:5]:>5}{atom_id_safe:5d}{coords[i][0] / 10.0:8.3f}{coords[i][1] / 10.0:8.3f}{coords[i][2] / 10.0:8.3f}\n")
        f.write(f"{box_tensor[0] / 10.0:10.5f}{box_tensor[4] / 10.0:10.5f}{box_tensor[8] / 10.0:10.5f}\n")


def write_atom_types_itp(path, unique_types):
    with open(path, 'w') as f:
        f.write("[ atomtypes ]\n")
        f.write("; name   bond_type     mass       charge   ptype          sigma        epsilon\n")
        for b_type, o_atom in unique_types.items():
            f.write(
                f"  {b_type:<8} {o_atom.bond_type:<8} {o_atom.mass:>10.4f} {o_atom.charge:>8.4f} {o_atom.ptype:<3} {o_atom.sigma:>14.6e} {o_atom.epsilon:>14.6e}\n")


def write_single_molecule_itp(path, rdmol, mol_name, res_id, res_names, p_atom, p_bonded, p_improper):
    bonds = [v for k, v in p_bonded.items() if len(k) == 2]
    angles = [v for k, v in p_bonded.items() if len(k) == 3]
    dihedrals = [v for k, v in p_bonded.items() if len(k) == 4]
    cgrp = 0

    with open(path, 'w') as f:
        f.write(f"[ moleculetype ]\n; Name            nrexcl\n{mol_name:<16}  3\n\n")
        f.write("[ atoms ]\n;   nr       type  resnr residue  atom   cgnr     charge       mass\n")
        for atom in rdmol.GetAtoms():
            idx = atom.GetIdx()
            o_atom = p_atom[idx]
            f.write(
                f"{idx+1:>6} {o_atom.bond_type:>10} {res_id[idx]:>6} {res_names[idx][:5]:>6} {f'A{idx}'[:5]:>6} {cgrp:>6} {o_atom.charge:>10.4f} {o_atom.mass:>10.4f}\n")

        if bonds:
            f.write("\n[ bonds ]\n")
            for b in bonds: f.write(f"{b.indices[0]+1:>5} {b.indices[1]+1:>5} {b.ftype:>5} {b.r0:>14.6e} {b.k:>14.6e}\n")
        if angles:
            f.write("\n[ angles ]\n")
            for a in angles: f.write(
                f"{a.indices[0]+1:>5} {a.indices[1]+1:>5} {a.indices[2]+1:>5} {a.ftype:>5} {a.t0:>14.6e} {a.k:>14.6e}\n")
        if dihedrals:
            f.write("\n[ dihedrals ]\n")
            for d in dihedrals: f.write(
                f"{d.indices[0]+1:>5} {d.indices[1]+1:>5} {d.indices[2]+1:>5} {d.indices[3]+1:>5} {d.ftype:>5} {d.c0:>13.5e} {d.c1:>13.5e} {d.c2:>13.5e}\n")
        if p_improper:
            f.write("\n[ dihedrals ] ; Improper\n")
            for imp in p_improper.values():
                p_str = " ".join(f"{x:>14.6e}" for x in imp.params)
                f.write(
                    f"{imp.indices[0]+1:>5} {imp.indices[1]+1:>5} {imp.indices[2]+1:>5} {imp.indices[3]+1:>5} {imp.ftype:>5} {p_str}\n")


def write_master_top_file(path, include_lines, scheme):
    with open(path, 'w') as f:
        f.write("; ==================================================================\n")
        f.write("; Master Topology total control file for GROMACS simulation\n")
        f.write("; ==================================================================\n\n")
        f.write(
            "[ defaults ]\n; nbfunc        comb-rule       gen-pairs       fudgeLJ         fudgeQQ\n1             3               yes             0.5             0.5\n\n")
        f.write('#include "atom_types.itp"\n')
        for line in include_lines:
            f.write(line + "\n")
        f.write("\n[ system ]\nAutomated Combined Massive Assembly System\n\n[ molecules ]\n; Compound        #mols\n")
        for mol_name, count in scheme:
            f.write(f"{mol_name:<16}  {count}\n")

parse_and_split_system('test_data/test_system.sdf', "test_split_output")
run_massive_system_pipeline('test_split_output/split_mols_fixed.sdf', 'test_split_output')
