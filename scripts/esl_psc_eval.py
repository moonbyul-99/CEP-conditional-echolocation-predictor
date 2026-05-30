#!/usr/bin/env python3
"""
ESL-PSC 留一预测评估脚本

对全部 104 个物种执行 ESL-PSC 留一预测，保存结果 CSV。
ESL-PSC 内部使用多进程并行训练单个模型（每个 ESL 模型一个进程），
外层串行处理物种，避免多进程嵌套。

依赖：pip install group-lasso

用法：
    cd CEP_project

    # 全量 104 物种（默认 64 进程用于内部 ESL 并行）
    python scripts/esl_psc_eval.py

    # 自定义内部并行数
    python scripts/esl_psc_eval.py --n-cpu 32

    # 单物种详细日志
    python scripts/esl_psc_eval.py --species Desmodus_rotundus --verbose
"""

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 项目路径设置
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    LEAVE_ONE_DIR,
    MSA_DF_DIR,
    FEATURE_DATA_DIR,
    RESULTS_DIR,
    N_CPU,
)

from src.esl import (
    one_hot_encode_sequences,
    ESLPSCClassifier,
)

# ---------------------------------------------------------------------------
# 全量特征矩阵缓存（parquet 路径，由主进程生成）
# ---------------------------------------------------------------------------
FULL_FEATURE_PARQUET = os.path.join(FEATURE_DATA_DIR, "esl_psc_full_feature.parquet")


# =====================================================================
#  阶段 1：主进程构建特征矩阵（从 716 CSV 拼接 + 存 parquet）
# =====================================================================


def _load_csv(path):
    return pd.read_csv(path, index_col=0)


def build_and_cache_feature_matrix(csv_dir, parquet_path, n_cpu=64):
    """从 CSV 文件重建完整特征矩阵并缓存为 parquet。"""
    csv_files = sorted([
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith('.csv')
    ])
    print(f"重建特征矩阵：并行读取 {len(csv_files)} 个 CSV 文件...")
    with Pool(processes=min(n_cpu, 64)) as pool:
        dfs = list(tqdm(
            pool.imap(_load_csv, csv_files),
            total=len(csv_files),
            desc="  Loading CSVs",
        ))
    feature_df = pd.concat(dfs, axis=1)
    print(f"  完整特征矩阵: {feature_df.shape}")

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    # 转置存：features 做行（~728K）、species 做列（192）。
    # 避免 728K 列名塞进 parquet schema 触发 pyarrow thrift 溢出。
    feature_df.T.reset_index().to_parquet(parquet_path, index=False)
    print(f"  已缓存至: {parquet_path}")
    return feature_df


# =====================================================================
#  阶段 2：单物种 ESL-PSC 计算（可被多进程调用）
# =====================================================================


def parse_float_list(s):
    """解析逗号分隔的浮点数列表，如 '0.01,0.03,0.05'"""
    return [float(x.strip()) for x in s.split(',')]


def process_one_species(species_id, base_dir, full_feature_df,
                        lambda1_list, lambda2_list,
                        top_pct, n_alternates, n_cpu=1,
                        n_top_per_gene=50, verbose=False):
    """
    对单个物种运行 ESL-PSC 预测。

    Parameters
    ----------
    species_id : str
    base_dir : str
        103_leave 数据目录。
    full_feature_df : pd.DataFrame
        完整特征矩阵（species × gene_pos）。
    lambda1_list : list[float]
    lambda2_list : list[float]
    top_pct : float
        按 MFS 保留 top 比例模型做集成。
    n_alternates : int
        每个 order 内最多配对数。
    n_cpu : int
        ESL-PSC 内部并行训练的进程数。
    n_top_per_gene : int
        每个基因取 NMI 最高的位点数。
    verbose : bool
        是否输出逐阶段耗时日志和模型详情。

    Returns
    -------
    list : [species_id, label, species_cn, esl_psc_pred, esl_psc_proba]
           or (species_id, error_str) on failure
    """
    try:
        t0 = time.time()

        dir_path = os.path.join(base_dir, species_id)
        feature_df = pd.read_csv(os.path.join(dir_path, 'df_feature.csv'), index_col=0)
        meta_df = pd.read_csv(os.path.join(dir_path, 'df_meta.csv'), index_col=0)
        summary_df = pd.read_csv(os.path.join(dir_path, 'df_summary.csv'), index_col=0)
        t_load = time.time()

        assert (meta_df.index == feature_df.index).all()
        assert (summary_df.index == feature_df.columns).all()

        all_species = meta_df.index.tolist()
        train_species = [s for s in all_species if s != species_id]

        # 每个基因取 NMI 最高的 n_top_per_gene 个位点
        summary_df_copy = summary_df.copy()
        summary_df_copy['gene'] = [e.split('_')[0] for e in summary_df_copy.index]
        summary_top = (summary_df_copy
                       .groupby('gene', group_keys=False)
                       .apply(lambda g: g.nlargest(n_top_per_gene, 'NMI'),
                              include_groups=False))
        top_features = summary_top.index.tolist()

        # 从完整矩阵中取对应特征（full_feature_df: species × features）
        seq_df = full_feature_df.loc[all_species, top_features]
        t_select = time.time()

        # ---- ESL-PSC ----
        clf2 = ESLPSCClassifier(
            lambda1_list=lambda1_list, lambda2_list=lambda2_list,
            top_pct=top_pct, n_alternates=n_alternates,
            order_col="order_chinese_new", label_col="label",
            n_cpu=n_cpu, verbose=verbose,
        )
        clf2.fit(seq_df.loc[train_species], meta_df.loc[train_species],
                 max_iter=100, tol=1e-4)
        t_psc = time.time()

        psc_pred, psc_proba, psc_sps = clf2.predict_species(seq_df, species_id)
        t_total = time.time()

        if verbose:
            print(f"\n  [{species_id}] 总耗时 {t_total - t0:.1f}s | "
                  f"加载={t_load - t0:.1f}s 特征选择={t_select - t_load:.1f}s "
                  f"PSC({len(clf2.models_)}models)={t_psc - t_select:.1f}s")

        true_label = int(meta_df.loc[species_id, 'label'])
        species_cn = meta_df.loc[species_id, 'species_chinese']

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  [{species_id}]  === ESL-PSC 评估详情 ===")
            print(f"  中文名: {species_cn}")
            print(f"  真实标签: {true_label} (1=回声定位)")
            print(f"{'=' * 60}")
            print(f"  [数据加载] 耗时 {t_load - t0:.1f}s")
            print(f"  [特征选择] 耗时 {t_select - t_load:.1f}s")
            print(f"    策略: 每基因取 NMI 最高的 top {n_top_per_gene} 位点")
            print(f"  [ESL-PSC 训练] 耗时 {t_psc - t_select:.1f}s")
            print(f"    lambda1_list={lambda1_list}, lambda2_list={lambda2_list}")
            print(f"    top_pct={top_pct}, n_alternates={n_alternates}")
            print(f"    模型数: {len(clf2.models_)}")
            print(f"  [预测] 总耗时 {t_total - t0:.1f}s")
            print(f"    预测标签: {int(psc_pred)}, 回声概率: {float(psc_proba[0, 1] if psc_proba.ndim > 1 else psc_proba):.4f}")

        return [species_id, true_label, species_cn,
                int(psc_pred),
                float(psc_proba[0, 1] if psc_proba.ndim > 1 else psc_proba)]

    except Exception as e:
        return (species_id, str(e))


def main():
    parser = argparse.ArgumentParser(description='ESL-PSC 留一预测评估')
    parser.add_argument('--base-dir', type=str, default=LEAVE_ONE_DIR,
                        help=f'103_leave 数据目录（默认：{LEAVE_ONE_DIR}）')
    parser.add_argument('--csv-dir', type=str, default=MSA_DF_DIR,
                        help=f'CSV 特征文件目录（默认：{MSA_DF_DIR}）')
    parser.add_argument('--top-per-gene', type=int, default=100,
                        help='每个基因取 NMI 最高的位点数（默认：100）')
    parser.add_argument('--lambda1-list', type=str, default='0.03',
                        help='lambda1 值列表，逗号分隔（默认：0.03）')
    parser.add_argument('--lambda2-list', type=str, default='0.01',
                        help='lambda2 值列表，逗号分隔（默认：0.01）')
    parser.add_argument('--top-pct', type=float, default=0.3,
                        help='按 MFS 保留 top 比例模型做集成（默认：0.3）')
    parser.add_argument('--n-alternates', type=int, default=2,
                        help='每个 order 内最多配对数（默认：2）')
    parser.add_argument('--species', nargs='+', default=None,
                        help='指定物种列表进行快速测试（默认：全部 104 物种）')
    parser.add_argument('--verbose', action='store_true',
                        help='输出每个物种的详细日志')
    parser.add_argument('--n-cpu', type=int, default=N_CPU,
                        help=f'ESL-PSC 内部并行进程数（默认：{N_CPU}）')
    args = parser.parse_args()

    lambda1_list = parse_float_list(args.lambda1_list)
    lambda2_list = parse_float_list(args.lambda2_list)

    # ------------------------------------------------------------------
    #  1. 构建并缓存全量特征矩阵（如已缓存则复用）
    # ------------------------------------------------------------------
    if os.path.exists(FULL_FEATURE_PARQUET):
        print(f"全量特征矩阵缓存已存在：{FULL_FEATURE_PARQUET}")
        print(f"  如需重建请手动删除该文件后重新运行。")
    else:
        print("未找到特征矩阵缓存，开始构建...")
        build_and_cache_feature_matrix(args.csv_dir, FULL_FEATURE_PARQUET, n_cpu=args.n_cpu)

    # 加载特征矩阵
    print("加载特征矩阵...")
    df = pd.read_parquet(FULL_FEATURE_PARQUET)
    feature_col = df.columns[0]
    full_feature_df = df.set_index(feature_col).T
    full_feature_df.index.name = None

    # ------------------------------------------------------------------
    #  2. 获取物种列表
    # ------------------------------------------------------------------
    if args.species:
        species_list = args.species
    else:
        species_list = sorted([
            d for d in os.listdir(args.base_dir)
            if os.path.isdir(os.path.join(args.base_dir, d))
        ])
    n_species = len(species_list)
    print(f"\nESL-PSC 评估：{n_species} 个物种（内部并行 {args.n_cpu} 进程）")
    print(f"lambda1_list={lambda1_list}, lambda2_list={lambda2_list}, "
          f"top_pct={args.top_pct}, n_alternates={args.n_alternates}")

    # ------------------------------------------------------------------
    #  3. 串行处理每个物种（ESL-PSC 内部多进程并行）
    # ------------------------------------------------------------------
    results = []
    errors = []
    for species_id in tqdm(species_list, desc="Species"):
        res = process_one_species(
            species_id,
            base_dir=args.base_dir,
            full_feature_df=full_feature_df,
            lambda1_list=lambda1_list,
            lambda2_list=lambda2_list,
            top_pct=args.top_pct,
            n_alternates=args.n_alternates,
            n_cpu=args.n_cpu,
            n_top_per_gene=args.top_per_gene,
            verbose=args.verbose,
        )
        if isinstance(res, tuple) and len(res) == 2:
            errors.append(res)
            tqdm.write(f"  [ERROR] {res[0]}: {res[1]}")
        else:
            results.append(res)

    # ------------------------------------------------------------------
    #  4. 保存结果
    # ------------------------------------------------------------------
    columns = ['species', 'label', 'species_cn',
               'esl_psc_label', 'esl_psc_eco_prob']
    df = pd.DataFrame(results, columns=columns)
    df = df.set_index('species')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'esl_psc_eval.csv')
    df.to_csv(out_path)
    print(f"\n结果已保存：{out_path}")

    # ------------------------------------------------------------------
    #  5. 汇总准确率
    # ------------------------------------------------------------------
    if len(df) > 0:
        acc = (df['esl_psc_label'] == df['label']).mean()
        n_correct = (df['esl_psc_label'] == df['label']).sum()
        print(f"ESL-PSC: {n_correct}/{len(df)} = {acc:.4f}")

    if errors:
        print(f"\n{len(errors)} 个物种评估失败:")
        for sp, err in errors:
            print(f"  {sp}: {err}")


if __name__ == '__main__':
    main()
    # main()
