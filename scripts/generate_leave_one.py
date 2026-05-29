#!/usr/bin/env python3
"""
生成 103_leave 预计算数据

Pipeline（从 CSV 文件直接生成，无需依赖 hqy_new/ 目录）：
1. 读取所有基因的 CSV 文件，拼接为完整特征矩阵（species × all_sites）
2. 计算全局特征排序（NMI + eco_mutation + coverage + score）
3. 对每个物种执行 leave-one-out 特征选择，生成 103_leave/{species_id}/ 数据

用法：
    cd CEP_project
    python scripts/generate_leave_one.py \
        --csv-dir data/msa_df_717 \
        --metadata data/metadata/metadata_1.csv \
        --output-dir data/leave_one \
        --top-k 20000 \
        --n-cpu 64
"""

import argparse
import os
import sys
import functools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

# ---------------------------------------------------------------------------
# 项目路径设置
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import MSA_DF_DIR, METADATA_1_CSV, N_CPU

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SCORE_MAP_4 = {4: 1, 3: 0.95, 2: 0.9, 1: 0.1, 0: 0}
SCORE_MAP_5 = {5: 1, 4: 0.9, 3: 0.75, 2: 0.5, 1: 0.1, 0: 0}
ECO_ORDERS = ['啮齿目', '真盲缺目', '非洲兽目', '翼手目', '鲸目']  # 对齐 07_esl_compare Cell 12 顺序


# ===========================================================================
# Gap A: CSV → 完整特征矩阵
# ===========================================================================

def _load_csv(path):
    """读取单个 CSV 文件（模块级函数，可被 Pool pickle）"""
    return pd.read_csv(path, index_col=0)


def build_feature_matrix(csv_dir, n_cpu=64):
    """
    读取 csv_dir 下所有 CSV 文件，拼接为 (species, all_sites) DataFrame。
    
    等价于原始代码：
        file_list = os.listdir('data_717/data')
        res_list = []
        for file in tqdm(file_list):
            df = pd.read_csv(path, index_col=0)
            res_list.append(df)
        df = pd.concat(res_list, axis=1)
    
    Returns
    -------
    feature_df : pd.DataFrame
        shape = (n_species, n_sites)
    """
    csv_files = sorted([
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith('.csv')
    ])
    if not csv_files:
        raise FileNotFoundError(f"未在 {csv_dir} 中找到 CSV 文件。")

    print(f"读取 {len(csv_files)} 个 CSV 文件...")

    with Pool(processes=min(n_cpu, 64)) as pool:
        dfs = list(tqdm(
            pool.imap(_load_csv, csv_files),
            total=len(csv_files),
            desc="Loading CSVs"
        ))

    feature_df = pd.concat(dfs, axis=1)
    print(f"完整特征矩阵: {feature_df.shape[0]} 物种 × {feature_df.shape[1]} 位点")
    return feature_df


# ===========================================================================
# Gap B: 全局特征排序（summary_df）
# ===========================================================================

def _compute_nmi_chunk(args):
    """
    计算一个 chunk 中所有特征的 NMI 和 eco_mutation。
    输入: (col_names, feat_values, eco_feat_values, label_array)
    """
    col_names, feat_values, eco_feat_values, label = args
    nmi_list = []
    mode_list = []

    for i, col in enumerate(col_names):
        x = feat_values[:, i]
        nmi = normalized_mutual_info_score(label, x)
        nmi_list.append(nmi)

        # eco_mutation: 回声物种中的众数
        eco_col = eco_feat_values[:, i]
        if len(eco_col) > 0:
            # 使用 pd.Series.mode() 保持与原代码一致
            mode_val = pd.Series(eco_col).mode()
            mode_val = mode_val.iloc[0] if not mode_val.empty else np.nan
        else:
            mode_val = np.nan
        mode_list.append(mode_val)

    return col_names, nmi_list, mode_list


def compute_global_summary(feature_df, meta_df, n_cpu=64):
    """
    计算全局特征排序：NMI + eco_mutation + coverage + score。
    
    等价于 paper_result/01_figure_103.ipynb Cell 3,5,9,11。
    
    Returns
    -------
    summary_df : pd.DataFrame
        index = feature_name, columns = [NMI, Eco_Mode, gene, 非洲兽目, 鲸目, ...]
    """
    label = meta_df['label'].values
    eco_idx = label == 1
    eco_feature = feature_df.iloc[eco_idx, :]

    # 切分列
    col_names = feature_df.columns.tolist()
    n_cols = len(col_names)
    max_workers = min(n_cpu, os.cpu_count() or 1, n_cols)
    chunk_size = max(1, n_cols // max_workers)

    chunks = [
        col_names[i:i + chunk_size]
        for i in range(0, n_cols, chunk_size)
    ]

    tasks = []
    for cols in chunks:
        feat_sub = feature_df[cols].values
        eco_sub = eco_feature[cols].values
        tasks.append((cols, feat_sub, eco_sub, label))

    print(f"NMI 计算: {max_workers} 进程, {len(tasks)} 个 chunks")

    nmi_dict = {}
    mode_dict = {}

    with Pool(processes=max_workers) as pool:
        for cols, nmi_vals, mode_vals in tqdm(
            pool.imap(_compute_nmi_chunk, tasks),
            total=len(tasks),
            desc="Computing NMI"
        ):
            for col, nmi, mv in zip(cols, nmi_vals, mode_vals):
                nmi_dict[col] = nmi
                mode_dict[col] = mv

    nmi_list = [nmi_dict[col] for col in col_names]
    eco_mutation = [mode_dict[col] for col in col_names]

    summary_df = pd.DataFrame({
        'NMI': nmi_list,
        'Eco_Mode': eco_mutation
    }, index=col_names)

    # 添加 gene 列
    summary_df['gene'] = [col.split('_')[0] for col in col_names]

    # 计算 coverage（5 个回声谱系）
    eco_mu = summary_df['Eco_Mode'].values
    feature_df_sorted = feature_df.loc[:, summary_df.index]

    res = []
    for order in ECO_ORDERS:
        idx = np.logical_and(
            meta_df['order_chinese_new'].values == order,
            meta_df['label'].values == 1
        )
        sub_feature = feature_df_sorted.loc[idx, :]
        if sub_feature.shape[0] == 0:
            x = np.zeros(len(eco_mu))
        else:
            x = (sub_feature.values == eco_mu).sum(axis=0) / sub_feature.shape[0]
        res.append(x)

    res = np.array(res)
    cover_df = pd.DataFrame(res.T, index=summary_df.index, columns=ECO_ORDERS)
    summary_df = pd.concat([summary_df, cover_df], axis=1)

    # 计算 eco_cover 和 score
    summary_df['eco_cover'] = (summary_df[ECO_ORDERS] > 0).sum(axis=1)
    map_dic = {5: 1, 4: 0.9, 3: 0.75, 2: 0.5, 1: 0.1, 0: 0}
    summary_df['eco_cover_score'] = summary_df['eco_cover'].map(map_dic)
    summary_df['score'] = summary_df['NMI'] * summary_df['eco_cover_score']
    summary_df = summary_df.sort_values(by='score', ascending=False)

    print(f"全局特征排序完成: {len(summary_df)} 个位点")
    return summary_df


# ===========================================================================
# Gap C: 生成 103_leave 数据
# ===========================================================================

def generate_one_species(species_id, feature_df_full, meta_df_full, 
                         global_top_features, output_dir):
    """
    对单个物种执行 leave-one-out 特征选择。
    
    等价于 notebook/07_esl_compare.ipynb Cell 12。
    
    Parameters
    ----------
    species_id : str
    feature_df_full : pd.DataFrame
        完整特征矩阵 (all_species, top_k_features)
    meta_df_full : pd.DataFrame
        元数据（已过滤 label != 2）
    global_top_features : list
        全局 top K 特征名
    output_dir : str
    """
    save_dir = os.path.join(output_dir, species_id)
    os.makedirs(save_dir, exist_ok=True)

    # Leave-one-out: 移除当前物种
    train_species = [s for s in meta_df_full.index if s != species_id]
    feature_df = feature_df_full.loc[train_species, :]
    meta_df = meta_df_full.loc[train_species, :]

    # 计算 NMI
    y = meta_df['label'].values
    nmi_list = []
    for i in range(feature_df.shape[1]):
        nmi = normalized_mutual_info_score(y, feature_df.iloc[:, i].values)
        nmi_list.append(nmi)

    # 获取 eco_mutation
    eco_idx = meta_df['label'].values == 1
    eco_feature = feature_df.iloc[eco_idx, :]
    eco_mutation = eco_feature.mode(axis=0).iloc[0, :].values

    # 获取 coverage（4 或 5 支）
    cover_res = {}
    eco_meta = meta_df.loc[eco_idx, :]
    clade_num = 5

    for key in ECO_ORDERS:
        if key not in eco_meta['order_chinese_new'].values:
            cover_res[key] = np.zeros(len(eco_mutation))
            clade_num = 4
        else:
            idx = eco_meta['order_chinese_new'] == key
            cover_res[key] = (eco_feature.loc[idx, :] == eco_mutation).sum(axis=0) / idx.sum()

    cover_df = pd.DataFrame(cover_res, index=feature_df.columns)
    cover_df['NMI'] = nmi_list
    cover_df['Eco_Mode'] = eco_mutation
    cover_df['eco_cover'] = (cover_df[ECO_ORDERS] > 0).sum(axis=1)

    if clade_num == 4:
        cover_df['cover_score'] = cover_df['eco_cover'].map(SCORE_MAP_4)
    else:
        cover_df['cover_score'] = cover_df['eco_cover'].map(SCORE_MAP_5)

    cover_df['score'] = cover_df['cover_score'] * cover_df['NMI']
    cover_df = cover_df.sort_values(by='score', ascending=False)

    # 取 top 10000 保存
    cover_df = cover_df.iloc[:10000, :]

    # 保存（对齐原始 notebook：feature_df 和 meta_df 保留全部有效物种，
    # 评估时需要访问被测物种的特征做预测）
    cover_df.to_csv(os.path.join(save_dir, 'df_summary.csv'))
    feature_df_full.loc[meta_df_full.index, cover_df.index].to_csv(
        os.path.join(save_dir, 'df_feature.csv'))
    meta_df_full.to_csv(os.path.join(save_dir, 'df_meta.csv'))

    return species_id


def generate_leave_one_data(feature_df, meta_df, summary_df, output_dir, 
                            top_k=20000, n_cpu=64):
    """
    对全部物种生成 103_leave 数据。
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        完整特征矩阵
    meta_df : pd.DataFrame
        元数据
    summary_df : pd.DataFrame
        全局特征排序
    output_dir : str
    top_k : int
        全局预筛选的特征数（默认 20000）
    n_cpu : int
    """
    # 取全局 top K 特征
    top_features = summary_df.sort_values(by='score', ascending=False).index[:top_k].tolist()
    feature_df_subset = feature_df.loc[:, top_features]

    # 过滤 label != 2
    valid_idx = meta_df['label'].values != 2
    meta_df_valid = meta_df.loc[valid_idx, :]
    species_list = meta_df_valid.index.tolist()

    print(f"\n生成 103_leave 数据: {len(species_list)} 个物种")
    print(f"全局预筛选: top {top_k} 特征")
    print(f"输出目录: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 多进程
    task_func = functools.partial(
        generate_one_species,
        feature_df_full=feature_df_subset,
        meta_df_full=meta_df_valid,
        global_top_features=top_features,
        output_dir=output_dir
    )

    results = []
    with Pool(processes=min(n_cpu, len(species_list))) as pool:
        for res in tqdm(
            pool.imap_unordered(task_func, species_list),
            total=len(species_list),
            desc="Generating 103_leave"
        ):
            results.append(res)

    print(f"\n完成! 生成 {len(results)} 个物种的预计算数据")


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='生成 103_leave 预计算数据（从 CSV 文件直接生成）'
    )
    parser.add_argument(
        '--csv-dir', type=str, default=MSA_DF_DIR,
        help=f'CSV 文件目录（默认：{MSA_DF_DIR}）'
    )
    parser.add_argument(
        '--metadata', type=str, default=METADATA_1_CSV,
        help=f'元数据文件（默认：{METADATA_1_CSV}）'
    )
    parser.add_argument(
        '--output-dir', type=str, 
        default=os.path.join(PROJECT_ROOT, 'data', 'leave_one'),
        help='输出目录（默认：data/leave_one）'
    )
    parser.add_argument(
        '--top-k', type=int, default=20000,
        help='全局预筛选特征数（默认：20000）'
    )
    parser.add_argument(
        '--save-summary', action='store_true',
        help='同时保存全局 summary_df.csv 到 results/'
    )
    parser.add_argument(
        '--n-cpu', type=int, default=N_CPU,
        help=f'并行进程数（默认：{N_CPU}）'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CEP 103_leave 预计算数据生成")
    print("=" * 60)

    # Step 1: 构建完整特征矩阵
    feature_df = build_feature_matrix(args.csv_dir, n_cpu=args.n_cpu)

    # Step 2: 加载元数据
    meta_df = pd.read_csv(args.metadata, index_col=0)
    print(f"元数据: {meta_df.shape[0]} 物种")

    # 对齐物种
    common_species = feature_df.index.intersection(meta_df.index)
    if len(common_species) < len(meta_df):
        missing = set(meta_df.index) - set(common_species)
        print(f"警告: {len(missing)} 个物种在 CSV 中未找到: {list(missing)[:5]}...")
    feature_df = feature_df.loc[common_species, :]
    meta_df = meta_df.loc[common_species, :]

    # Step 3: 计算全局特征排序
    print("\n" + "=" * 60)
    print("计算全局特征排序...")
    print("=" * 60)
    
    valid_idx = meta_df['label'].values != 2
    meta_df_valid = meta_df.loc[valid_idx, :]
    feature_df_valid = feature_df.loc[valid_idx, :]
    
    summary_df = compute_global_summary(feature_df_valid, meta_df_valid, n_cpu=args.n_cpu)

    if args.save_summary:
        from src.config import RESULTS_DIR
        os.makedirs(RESULTS_DIR, exist_ok=True)
        summary_path = os.path.join(RESULTS_DIR, 'summary_df.csv')
        summary_df.to_csv(summary_path)
        print(f"全局排序已保存: {summary_path}")

    # Step 4: 生成 103_leave 数据
    print("\n" + "=" * 60)
    print("生成 103_leave 数据...")
    print("=" * 60)
    
    generate_leave_one_data(
        feature_df, meta_df, summary_df, 
        args.output_dir, args.top_k, args.n_cpu
    )

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\n输出目录: {args.output_dir}")
    print(f"物种数: {len(os.listdir(args.output_dir))}")
    print("\n下一步: python scripts/leave_one_run.py --top-k 500")


if __name__ == '__main__':
    main()
