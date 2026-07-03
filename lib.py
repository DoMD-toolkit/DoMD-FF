from itertools import combinations
import logging
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



def print_opls_stats(forcefield, logger, level=logging.WARNING, md_mode=False):
    """
    Print force-field parameter coverage statistics.
    
    Parameters
    ----------
    forcefield : object
        Force-field object containing forcefield._meta.
    logger : object
        Python logger-like object.
    level : int or str
        Logging level.
    md_mode : bool
        If True, outputs a stripped-down, left-aligned plain text block.
    """
    if isinstance(level, str):
        level_name = level.upper()
        level = logging._nameToLevel.get(level_name)
        if not isinstance(level, int):
            raise ValueError(f"Invalid logging level: {level_name}")

    def emit(message: str):
        logger.log(level, message)

    meta = getattr(forcefield, "_meta", None)
    
    if not isinstance(meta, dict):
        if md_mode:
            emit("\n".join([
                "FORCE FIELD PARAMETERIZATION STATISTICS",
                "WARNING: forcefield._meta is missing or invalid."
            ]))
        else:
            emit("\n" + "\n".join([
                " ================================================================",
                "              FORCE FIELD PARAMETERIZATION STATISTICS             ",
                " ================================================================",
                "   WARNING: forcefield._meta is missing or invalid.",
                " ================================================================"
            ]))
        return

    rows = [
        ("ATOMS",     "n_atom", "t_atom"),
        ("BONDS",     "n_bond", "t_bond"),
        ("ANGLES",    "n_ang",  "t_ang"),
        ("DIHEDRALS", "n_dih",  "t_dih"),
        ("IMPROPERS", "n_imp",  "t_imp"),
    ]

    data_rows = []

    total_found = 0
    total_expected = 0

    for label, n_key, t_key in rows:
        found = int(meta.get(n_key, 0) or 0)
        total = int(meta.get(t_key, 0) or 0)
        missing = max(total - found, 0)

        total_found += found
        total_expected += total

        data_rows.append((label, found, total, missing))

    total_missing = max(total_expected - total_found, 0)
    pre_msg = "Force field parameterization success." if forcefield.success else "Force field parameterization failed."

    if md_mode:
        md_lines = [
            f"{pre_msg}\n",
            f"{'TERM':<10s} {'FOUND':>10s} {'TOTAL':>10s} {'MISSING':>10s}    {'COVERAGE':>8s}"
        ]
        
        for label, found, total, missing in data_rows:
            cov_str = f"{100.0 * found / total:>8.2f}%" if total > 0 else "     N/A"
            md_lines.append(f"{label:<10s} {found:10d} {total:10d} {missing:10d}   {cov_str}")
        
        tot_cov_str = f"{100.0 * total_found / total_expected:>8.2f}%" if total_expected > 0 else "     N/A"
        md_lines.append(f"{'TOTAL':<10s} {total_found:10d} {total_expected:10d} {total_missing:10d}   {tot_cov_str}")
        
        emit("\n".join(md_lines))

    else:
        lines = []
        lines.append(" ================================================================")
        lines.append("              FORCE FIELD PARAMETERIZATION STATISTICS             ")
        lines.append(" ================================================================")
        lines.append("")
        lines.append("   TERM        FOUND        TOTAL      MISSING     COVERAGE")
        lines.append(" ---------------------------------------------------------------")

        for label, found, total, missing in data_rows:
            coverage_str = f"{100.0 * found / total:8.2f}%" if total > 0 else "     N/A"
            lines.append(
                f"   {label:<10s} {found:10d} {total:10d} {missing:10d}   {coverage_str}"
            )

        lines.append(" ---------------------------------------------------------------")

        total_coverage_str = (
            f"{100.0 * total_found / total_expected:8.2f}%"
            if total_expected > 0 else
            "     N/A"
        )

        lines.append(
            f"   {'TOTAL':<10s} {total_found:10d} {total_expected:10d} "
            f"{total_missing:10d}   {total_coverage_str}"
        )
        lines.append(" ================================================================")
        
        emit(f"{pre_msg}\n" + "\n".join(lines))

