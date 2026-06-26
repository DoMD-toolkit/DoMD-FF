import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# ==========================================
# 1. 确保与你线上完全统一的 Hash 函数
# ==========================================
def atom_hash_func(mol: Chem.Mol, radius=6, n_bits=2048):
    """你统一后的第二个版本的 Hash 函数（返回元组）"""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()
    fp_with_info = fpgen.GetFingerprint(mol, additionalOutput=additional_output)
    bit_info = additional_output.GetBitInfoMap()

    atom_fps = {atom.GetIdx(): np.zeros(n_bits, dtype=int) for atom in mol.GetAtoms()}
    for bit_id, atom_info_list in bit_info.items():
        for atom_idx, _ in atom_info_list:
            atom_fps[atom_idx][bit_id] = 1

    return atom_fps, fp_with_info.ToBinary()


# ==========================================
# 2. 核心诊断搜索主函数
# ==========================================
def diagnose_tfsi_nitrogen(database_instance):
    print("=== [DEBUG START] 开始定向搜寻 TFSI 氮原子 ===")

    # 1. 严格使用你给的标准 TFSI SMILES 构建分子
    smiles = "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"
    mol = Chem.MolFromSmiles(smiles)

    # 💡 关键：根据你建库时是否有氢，决定是否 AddHs
    # 如果你建库时包含显式氢，请取消下面这行的注释：
    # mol = Chem.AddHs(mol)

    Chem.SanitizeMol(mol)

    # 2. 找到分子中唯一的 Nitrogen 原子
    n_atom = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            n_atom = atom
            break

    if n_atom is None:
        print("错误：未在输入的 SMILES 中找到氮原子！")
        return

    n_idx = n_atom.GetIdx()
    print(f"成功定位 Nitrogen 原子，在当前内存分子中的索引为 (Idx): {n_idx}")

    # 3. 计算并提取它的 atom_hash 数组
    hashes_dict, _ = atom_hash_func(mol, radius=6, n_bits=2048)
    n_atom_hash = hashes_dict[n_idx]


    # 4. 执行你的标准的打包装箱逻辑
    hash_str = np.packbits(n_atom_hash).tobytes()
    print(f"Nitrogen 的 hash_str (二进制表示): {hash_str}")
    # 打印该 Hash 的十六进制特征，方便去数据库里用 Navicat/SQLite Studio 直接手动 Select
    print(f"Nitrogen 的 hash_str (Hex 表示): {hash_str.hex()}")

    # 5. 调用你的数据库实例直接击穿搜索
    print("正在向数据库发送 search 请求...")
    res = database_instance.search('atom', hash_str=hash_str)

    # 6. 结果判定
    if res:
        print("🎉 [成功] 数据库中存在完全匹配的 TFSI 氮原子！")
        print(f"返回的原子数据为: {res}")
    else:
        print("❌ [失败] 数据库中没有找到这个氮原子！")
        print("原因可能为：")
        print("  1. 建库时的 TFSI 分子可能带了 H（变成了中性单质），导致 N 环境变了。")
        print("  2. 建库时的 TFSI 电荷画法不同，导致 N 上的 Formal Charge 不同。")
        print("  3. 数据库里存的根本不是这串 SMILES 生成的结构。")

    print("=== [DEBUG END] ===")


# ==========================================
# 3. 执行入口
# ==========================================
if __name__ == "__main__":
    # 请在这里传入你实例化好的、包含 OPLS 数据的真实 database 对象
    # from your_module import database
    import sys
    sys.path.append('E:\\downloads\\article\\high_throughput_system\\software\\DoMDv1.0.2\\DoMD-FF')
    from opls.opls_db.database import OplsDB
    opls_db = OplsDB('opls_small.db')
    diagnose_tfsi_nitrogen(opls_db)
    pass