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


def print_opls_stats(forcefield, logger, level=logging.WARNING):
    """
    Print force-field parameter coverage statistics in a Fortran-style block.

    Parameters
    ----------
    forcefield
        Force-field object containing forcefield._meta.
    logger
        Python logger-like object.
    level
        Logging level. Can be logging.INFO, logging.WARNING, etc.,
        or a string such as "info", "warning", "error".
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

    lines = []
    lines.append(" ================================================================")
    lines.append("              FORCE FIELD PARAMETERIZATION STATISTICS             ")
    lines.append(" ================================================================")
    lines.append("")
    lines.append("   TERM        FOUND        TOTAL      MISSING     COVERAGE")
    lines.append(" ---------------------------------------------------------------")

    total_found = 0
    total_expected = 0

    for label, n_key, t_key in rows:
        found = int(meta.get(n_key, 0) or 0)
        total = int(meta.get(t_key, 0) or 0)
        missing = max(total - found, 0)

        total_found += found
        total_expected += total

        coverage_str = f"{100.0 * found / total:8.2f}%" if total > 0 else "     N/A"

        lines.append(
            f"   {label:<10s} {found:10d} {total:10d} {missing:10d}   {coverage_str}"
        )

    lines.append(" ---------------------------------------------------------------")

    total_missing = max(total_expected - total_found, 0)
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
    pre_msg = "Force field parameterization success." if forcefield.success else "Force field parameterization failed."
    emit(f"{pre_msg}\n" + "\n".join(lines))
