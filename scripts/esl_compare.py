#!/usr/bin/env python3
"""
ESL / ESL-PSC 对比评估脚本

对全部 104 个物种执行 ESL 和 ESL-PSC 留一预测，保存结果 CSV。

等价于 notebook/07_esl_compare.ipynb Cell 3 + Cell 4 + Cell 6

依赖：pip install group-lasso

用法：
    cd CEP_project
    python scripts/esl_compare.py --n-cpu 32
"""

import argparse
import os
import sys
import traceback
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 项目路径设置
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import LEAVE_ONE_DIR, MSA_DF_DIR, RESULTS_DIR, N_CPU

from src.esl import (
    one_hot_encode_sequences,
    ESLClassifier,
    ESLPSCClassifier,
)


def _load_csv(path):
    return pd.read_csv(path, index_col=0)


def build_full_feature_matrix(csv_dir, n_cpu=64):
    """从 CSV 文件重建完整特征矩阵（等价于 data_raw.parquet）"""
    csv_files = sorted([
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith('.csv')
    ])
    print(f"重建特征矩阵：读取 {len(csv_files)} 个 CSV 文件...")
    with Pool(processes=min(n_cpu, 64)) as pool:
        dfs = list(tqdm(pool.imap(_load_csv, csv_files),
                        total=len(csv_files), desc="Loading"))
    # dfs: list of (species × positions), concat along columns → (species × all_positions)
    feature_df = pd.concat(dfs, axis=1)
    print(f"完整特征矩阵: {feature_df.shape}")
    return feature_df


def process_one_species(species_id, base_dir, full_feature_df, n_top_per_gene=100):
    """
    对单个物种运行 ESL + ESL-PSC 预测。

    Returns
    -------
    list : [species_id, label, species_cn, esl_pred, esl_proba, eslpsc_pred, eslpsc_proba]
           or (species_id, error_str) on failure
    """
    try:
        dir_path = os.path.join(base_dir, species_id)
        feature_df = pd.read_csv(os.path.join(dir_path, 'df_feature.csv'), index_col=0)
        meta_df = pd.read_csv(os.path.join(dir_path, 'df_meta.csv'), index_col=0)
        summary_df = pd.read_csv(os.path.join(dir_path, 'df_summary.csv'), index_col=0)

        assert (meta_df.index == feature_df.index).all()
        assert (summary_df.index == feature_df.columns).all()

        all_species = meta_df.index.tolist()
        train_species = [s for s in all_species if s != species_id]
        train_label = meta_df.loc[train_species, 'label'].values

        # 每个基因取 NMI 最高的 n_top_per_gene 个位点
        summary_df_copy = summary_df.copy()
        summary_df_copy['gene'] = [e.split('_')[0] for e in summary_df_copy.index]
        summary_top = (summary_df_copy
                       .groupby('gene', group_keys=False)
                       .apply(lambda g: g.nlargest(n_top_per_gene, 'NMI')))
        top_features = summary_top.index.tolist()

        # 从完整矩阵中取对应特征
        seq_df = full_feature_df.loc[top_features, all_species].T  # (species, features)

        # ---- ESL ----
        train_id = np.array([s != species_id for s in all_species])
        X, feat_names, group_ids, pos_names, gene_names = one_hot_encode_sequences(seq_df)
        X_train, X_test = X[train_id], X[~train_id]

        clf = ESLClassifier(lambda1=0.03, lambda2=0.01, tol=1e-4, max_iter=100)
        clf.fit(X_train, train_label, group_ids,
                feature_names=feat_names,
                position_names=pos_names,
                gene_names=gene_names)

        esl_pred = clf.predict(X_test)
        esl_proba = clf.predict_proba(X_test)

        # ---- ESL-PSC ----
        clf2 = ESLPSCClassifier(
            lambda1_list=[0.03], lambda2_list=[0.01],
            top_pct=0.3, n_alternates=2,
            order_col="order_chinese_new", label_col="label",
        )
        clf2.fit(seq_df.loc[train_species], meta_df.loc[train_species],
                 max_iter=100, tol=1e-4)

        psc_pred, psc_proba, psc_sps = clf2.predict_species(seq_df, species_id)

        true_label = int(meta_df.loc[species_id, 'label'])
        species_cn = meta_df.loc[species_id, 'species_chinese']

        return [species_id, true_label, species_cn,
                int(esl_pred[0]), float(esl_proba[0, 1]),
                int(psc_pred), float(psc_proba[0, 1] if psc_proba.ndim > 1 else psc_proba)]

    except Exception as e:
        return (species_id, str(e))


def main():
    parser = argparse.ArgumentParser(description='ESL / ESL-PSC 对比评估')
    parser.add_argument('--base-dir', type=str, default=LEAVE_ONE_DIR,
                        help=f'103_leave 数据目录（默认：{LEAVE_ONE_DIR}）')
    parser.add_argument('--csv-dir', type=str, default=MSA_DF_DIR,
                        help=f'CSV 特征文件目录（默认：{MSA_DF_DIR}）')
    parser.add_argument('--top-per-gene', type=int, default=100,
                        help='每个基因取 NMI 最高的位点数（默认：100）')
    parser.add_argument('--n-cpu', type=int, default=N_CPU,
                        help=f'并行进程数（默认：{N_CPU}）')
    args = parser.parse_args()

    # 重建完整特征矩阵
    full_feature_df = build_full_feature_matrix(args.csv_dir, n_cpu=args.n_cpu)

    # 获取物种列表
    species_list = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])
    print(f"\nESL/ESL-PSC 对比：{len(species_list)} 个物种")

    task_func = partial(process_one_species,
                        base_dir=args.base_dir,
                        full_feature_df=full_feature_df,
                        n_top_per_gene=args.top_per_gene)

    results = []
    errors = []

    # 串行执行（ESL 内部已多线程，避免嵌套）
    for sp in tqdm(species_list, desc="ESL/ESL-PSC"):
        res = task_func(sp)
        if isinstance(res, tuple) and len(res) == 2:
            errors.append(res)
        else:
            results.append(res)

    # 保存结果
    columns = ['species', 'label', 'species_cn',
               'esl_label', 'esl_eco_prob',
               'esl_psc_label', 'esl_psc_eco_prob']
    df = pd.DataFrame(results, columns=columns)
    df = df.set_index('species')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'esl_compare.csv')
    df.to_csv(out_path)
    print(f"\n结果已保存：{out_path}")

    # 汇总准确率
    for method, col in [('ESL', 'esl_label'), ('ESL-PSC', 'esl_psc_label')]:
        acc = (df[col] == df['label']).mean()
        n_correct = (df[col] == df['label']).sum()
        print(f"{method}: {n_correct}/{len(df)} = {acc:.4f}")

    if errors:
        print(f"\n{len(errors)} 个物种评估失败:")
        for sp, err in errors:
            print(f"  {sp}: {err}")


if __name__ == '__main__':
    main()
