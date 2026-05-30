#!/usr/bin/env python3
"""
ESL 留一预测评估脚本（多进程并行）

对全部 104 个物种执行 ESL 留一预测，保存结果 CSV。
每个物种分配一个独立进程（ESL 坐标下降为单线程任务，多进程接近线性加速）。

依赖：pip install group-lasso

用法：
    cd CEP_project

    # 全量 104 物种（默认 64 进程）
    python scripts/esl_eval.py

    # 自定义进程数
    python scripts/esl_eval.py --n-cpu 32

    # 单物种详细日志
    python scripts/esl_eval.py --species Desmodus_rotundus --verbose --top-per-gene 50
"""

import argparse
import os
import sys
import time
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

from src.config import (
    LEAVE_ONE_DIR,
    MSA_DF_DIR,
    FEATURE_DATA_DIR,
    RESULTS_DIR,
    N_CPU,
)

from src.esl import (
    one_hot_encode_sequences,
    ESLClassifier,
)

# ---------------------------------------------------------------------------
# 全量特征矩阵缓存（parquet 路径，由主进程生成，子进程读取）
# ---------------------------------------------------------------------------
FULL_FEATURE_PARQUET = os.path.join(FEATURE_DATA_DIR, "esl_full_feature.parquet")


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
        if f.endswith(".csv")
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
#  阶段 2：单物种 ESL 计算（可被多进程调用）
# =====================================================================


def process_one_species(
    species_id,
    base_dir,
    full_feature_df,
    n_top_per_gene=100,
    verbose=False,
):
    """
    对单个物种运行 ESL 预测。

    Parameters
    ----------
    species_id : str
    base_dir : str
        103_leave 数据目录。
    full_feature_df : pd.DataFrame
        完整特征矩阵（species × gene_pos）。多进程模式下，由 worker 自行从 parquet 加载。
    n_top_per_gene : int
        每个基因取 NMI 最高的位点数。
    verbose : bool
        是否输出逐阶段耗时日志和模型详情（仅单物种模式有效）。

    Returns
    -------
    list : [species_id, label, species_cn, esl_label, esl_eco_prob]
           or (species_id, error_str) on failure
    """
    try:
        t0 = time.time()

        # ---- 加载 leave_one 数据 ----
        dir_path = os.path.join(base_dir, species_id)
        feature_df = pd.read_csv(os.path.join(dir_path, "df_feature.csv"), index_col=0)
        meta_df = pd.read_csv(os.path.join(dir_path, "df_meta.csv"), index_col=0)
        summary_df = pd.read_csv(os.path.join(dir_path, "df_summary.csv"), index_col=0)
        t_load = time.time()

        assert (meta_df.index == feature_df.index).all()
        assert (summary_df.index == feature_df.columns).all()

        all_species = meta_df.index.tolist()
        train_label = meta_df.loc[
            [s for s in all_species if s != species_id], "label"
        ].values

        n_train_pos = int((train_label == 1).sum())
        n_train_neg = int((train_label == 0).sum())

        # ---- 特征选择：每基因 top N 位点 by NMI ----
        summary_df_copy = summary_df.copy()
        summary_df_copy["gene"] = [e.split("_")[0] for e in summary_df_copy.index]
        n_genes = summary_df_copy["gene"].nunique()
        summary_top = (
            summary_df_copy
            .groupby("gene", group_keys=False)
            .apply(
                lambda g: g.nlargest(n_top_per_gene, "NMI"),
                include_groups=False,
            )
        )
        top_features = summary_top.index.tolist()

        seq_df = full_feature_df.loc[all_species, top_features]
        t_select = time.time()

        # ---- one-hot 编码 ----
        X, feat_names, group_ids, pos_names, gene_names = one_hot_encode_sequences(seq_df)
        n_positions = len(pos_names)
        n_onehot = X.shape[1]
        n_groups = len(np.unique(group_ids)) if len(group_ids) > 0 else 0
        t_encode = time.time()

        # ---- ESL 训练 ----
        train_id = np.array([s != species_id for s in all_species])
        X_train, X_test = X[train_id], X[~train_id]
        n_train = X_train.shape[0]

        clf = ESLClassifier(lambda1=0.03, lambda2=0.01, tol=1e-4, max_iter=100)
        clf.fit(
            X_train, train_label, group_ids,
            feature_names=feat_names,
            position_names=pos_names,
            gene_names=gene_names,
        )
        t_esl = time.time()

        esl_pred = clf.predict(X_test)
        esl_proba = clf.predict_proba(X_test)
        t_total = time.time()

        # ---- 提取模型稀疏度信息 ----
        coef_nonzero = int((np.abs(clf.coef_) > 1e-10).sum())
        gss = clf.get_GSS()
        pss = clf.get_PSS()
        n_selected_genes = len([g for g, s in gss.items() if s > 0])
        n_selected_positions = len([p for p, s in pss.items() if s > 0])
        top_genes = list(gss.keys())[:5] if gss else []

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  [{species_id}]  === ESL 评估详情 ===")
            print(f"  中文名: {meta_df.loc[species_id, 'species_chinese']}")
            print(f"  真实标签: {int(meta_df.loc[species_id, 'label'])} (1=回声定位)")
            print(f"{'=' * 60}")
            print(f"  [数据加载] 耗时 {t_load - t0:.1f}s")
            print(f"    103_leave 目录特征总数: {summary_df.shape[0]}")
            print(f"    训练集物种: {n_train} (正样本={n_train_pos}, 负样本={n_train_neg})")
            print(f"  [特征选择] 耗时 {t_select - t_load:.1f}s")
            print(f"    策略: 每基因取 NMI 最高的 top {n_top_per_gene} 位点")
            print(f"    筛选后: {len(top_features)} 个原始位点, {n_genes} 个基因")
            print(f"  [One-hot 编码] 耗时 {t_encode - t_select:.1f}s")
            print(f"    原始位点数: {n_positions}")
            print(f"    One-hot 特征数: {n_onehot}")
            print(f"    Group 数 (基因): {n_groups}")
            print(f"  [ESL 训练] 耗时 {t_esl - t_encode:.1f}s")
            print(f"    lambda1={clf.lambda1}, lambda2={clf.lambda2}")
            print(f"    max_iter={clf.max_iter}, tol={clf.tol}")
            print(f"    非零系数: {coef_nonzero} / {n_onehot} "
                  f"({100.0 * coef_nonzero / max(n_onehot, 1):.1f}%)")
            print(f"    选中基因数: {n_selected_genes} / {n_groups}")
            print(f"    选中位点数: {n_selected_positions} / {n_positions}")
            if top_genes:
                print(
                    f"    Top 5 基因 (GSS): "
                    + ", ".join(f"{g}({s:.4f})" for g, s in list(gss.items())[:5])
                )
            print(f"  [预测] 总耗时 {t_total - t0:.1f}s")
            print(f"    SPS={clf.decision_function(X_test)[0]:.4f}")
            print(f"    预测标签: {int(esl_pred[0])}, 回声概率: {float(esl_proba[0, 1]):.4f}")

        true_label = int(meta_df.loc[species_id, "label"])
        species_cn = meta_df.loc[species_id, "species_chinese"]

        return [
            species_id, true_label, species_cn,
            int(esl_pred[0]), float(esl_proba[0, 1]),
        ]

    except Exception as e:
        return (species_id, str(e))


# =====================================================================
#  Worker：自包含地从 parquet 加载特征矩阵后计算
# =====================================================================


def _load_full_feature(parquet_path):
    """加载转置缓存的 parquet 并还原为 species × features 矩阵。"""
    df = pd.read_parquet(parquet_path)
    # 第一列是 feature 名（gene_pos），设为索引后转置还原
    feature_col = df.columns[0]
    df = df.set_index(feature_col).T
    df.index.name = None
    return df


def _worker_esl(species_id, base_dir, parquet_path, n_top_per_gene):
    """多进程 worker：加载 parquet → 调用 process_one_species。"""
    full_feature_df = _load_full_feature(parquet_path)
    return process_one_species(
        species_id,
        base_dir=base_dir,
        full_feature_df=full_feature_df,
        n_top_per_gene=n_top_per_gene,
        verbose=False,  # 多进程下不开 verbose，避免输出交错
    )


# =====================================================================
#  main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="ESL 留一预测评估（多进程并行）")
    parser.add_argument(
        "--base-dir", type=str, default=LEAVE_ONE_DIR,
        help=f"103_leave 数据目录（默认：{LEAVE_ONE_DIR}）",
    )
    parser.add_argument(
        "--csv-dir", type=str, default=MSA_DF_DIR,
        help=f"CSV 特征文件目录（默认：{MSA_DF_DIR}）",
    )
    parser.add_argument(
        "--top-per-gene", type=int, default=100,
        help="每个基因取 NMI 最高的位点数（默认：100）",
    )
    parser.add_argument(
        "--species", nargs="+", default=None,
        help="指定物种列表进行快速测试（默认：全部 104 物种）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="输出每个物种的详细日志（仅单物种或少量物种时有效）",
    )
    parser.add_argument(
        "--n-cpu", type=int, default=N_CPU,
        help=f"并行进程数（默认：{N_CPU}）",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  1. 构建并缓存全量特征矩阵（如已缓存则复用）
    # ------------------------------------------------------------------
    if os.path.exists(FULL_FEATURE_PARQUET):
        print(f"全量特征矩阵缓存已存在：{FULL_FEATURE_PARQUET}")
        print(f"  如需重建请手动删除该文件后重新运行。")
    else:
        print("未找到特征矩阵缓存，开始构建...")
        build_and_cache_feature_matrix(args.csv_dir, FULL_FEATURE_PARQUET, n_cpu=args.n_cpu)

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
    print(f"\nESL 评估：{n_species} 个物种")

    # ------------------------------------------------------------------
    #  3. 执行
    # ------------------------------------------------------------------
    if n_species == 1:
        # 单物种模式：串行 + verbose
        print("  单物种模式，串行执行...")
        full_feature_df = _load_full_feature(FULL_FEATURE_PARQUET)
        res = process_one_species(
            species_list[0],
            base_dir=args.base_dir,
            full_feature_df=full_feature_df,
            n_top_per_gene=args.top_per_gene,
            verbose=args.verbose,
        )
        raw_results = [res]
    else:
        # 多物种模式：多进程并行
        n_workers = min(args.n_cpu, n_species)
        print(f"  并行进程数：{n_workers}")

        worker_func = partial(
            _worker_esl,
            base_dir=args.base_dir,
            parquet_path=FULL_FEATURE_PARQUET,
            n_top_per_gene=args.top_per_gene,
        )
        with Pool(processes=n_workers) as pool:
            raw_results = list(tqdm(
                pool.imap(worker_func, species_list),
                total=n_species,
                desc="  ESL progress",
            ))

    # ------------------------------------------------------------------
    #  4. 分拣结果与错误
    # ------------------------------------------------------------------
    results = []
    errors = []
    for res in raw_results:
        if isinstance(res, tuple) and len(res) == 2:
            errors.append(res)
        else:
            results.append(res)

    # ------------------------------------------------------------------
    #  5. 保存结果
    # ------------------------------------------------------------------
    columns = ["species", "label", "species_cn", "esl_label", "esl_eco_prob"]
    df = pd.DataFrame(results, columns=columns)
    df = df.set_index("species")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "esl_eval.csv")
    df.to_csv(out_path)
    print(f"\n结果已保存：{out_path}")

    # ------------------------------------------------------------------
    #  6. 汇总准确率
    # ------------------------------------------------------------------
    if len(df) > 0:
        acc = (df["esl_label"] == df["label"]).mean()
        n_correct = (df["esl_label"] == df["label"]).sum()
        print(f"ESL: {n_correct}/{len(df)} = {acc:.4f}")

    if errors:
        print(f"\n{len(errors)} 个物种评估失败:")
        for sp, err in errors:
            print(f"  {sp}: {err}")


if __name__ == "__main__":
    main()
