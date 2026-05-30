#!/usr/bin/env python3
"""
多方法评估汇总脚本

读取所有方法的留一预测结果，计算指标，生成可视化热图与完整报告。

数据来源：
  - ESL:        results/esl_eval.csv
  - ESL-PSC:    results/esl_psc_eval.csv
  - CEP:        results/logs/cep_leave_one_*.csv（最新文件）
  - 消融:        results/ablation_study.csv
  - 四模型:      results/method_compare/test_pred_*.csv

用法：
    cd CEP_project
    python scripts/eval_summary.py
"""

import argparse
import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    METADATA_1_CSV,
    RESULTS_DIR,
    LOGS_DIR,
)

# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------
METHOD_COMPARE_DIR = os.path.join(RESULTS_DIR, "method_compare")

# 方法名 → 文件名 / 列名 映射
# ====== 消融变体说明 (ablation_study.py VARIANTS) ======
#   coldstart_score  — CEP 标准 RF 冷启动（对所有物种统一使用冷启动+RF）
#   local_no_cold    — CEP 无冷启动（本地 RF，同目训练集，无冷启动参考物种选择）
#   global_rf        — 全局 RF（无特征选择，无冷启动）
#   coldstart_random — CEP 无特征选择（冷启动+随机打乱特征）
#   coldstart_nmi    — CEP NMI-only（冷启动+仅NMI排序，不乘cover_score）
METHOD_CONFIG = {
    # ---- ESL & ESL-PSC ----
    "ESL": {
        "file": os.path.join(RESULTS_DIR, "esl_eval.csv"),
        "pred_col": "esl_label",
        "prob_col": "esl_eco_prob",
    },
    "ESL-PSC": {
        "file": os.path.join(RESULTS_DIR, "esl_psc_eval.csv"),
        "pred_col": "esl_psc_label",
        "prob_col": "esl_psc_eco_prob",
    },
    # ---- 消融变体 ----
    "CEP (coldstart)": {
        "file": os.path.join(RESULTS_DIR, "ablation_study.csv"),
        "pred_col": "coldstart_score_pred",
        "label_col": "true_label",
    },
    "CEP w/o coldstart": {
        "file": os.path.join(RESULTS_DIR, "ablation_study.csv"),
        "pred_col": "local_no_cold_pred",   # dict → 多数投票
        "label_col": "true_label",
        "is_dict": True,
    },
    "Global RF": {
        "file": os.path.join(RESULTS_DIR, "ablation_study.csv"),
        "pred_col": "global_rf_pred",        # dict → 多数投票
        "label_col": "true_label",
        "is_dict": True,
    },
    "CEP w/o featsel": {
        "file": os.path.join(RESULTS_DIR, "ablation_study.csv"),
        "pred_col": "coldstart_random_pred",
        "label_col": "true_label",
    },
    "CEP NMI-only": {
        "file": os.path.join(RESULTS_DIR, "ablation_study.csv"),
        "pred_col": "coldstart_nmi_pred",
        "label_col": "true_label",
    },
    # ---- 四模型（方法对比）----
    "Random Forest": {
        "file": os.path.join(METHOD_COMPARE_DIR, "test_pred_RandomForest.csv"),
        "train_acc": os.path.join(METHOD_COMPARE_DIR, "train_acc_RandomForest.csv"),
    },
    "Logistic Regression": {
        "file": os.path.join(METHOD_COMPARE_DIR, "test_pred_LogisticRegression.csv"),
        "train_acc": os.path.join(METHOD_COMPARE_DIR, "train_acc_LogisticRegression.csv"),
    },
    "Naive Bayes": {
        "file": os.path.join(METHOD_COMPARE_DIR, "test_pred_NaiveBayes(Categorical).csv"),
        "train_acc": os.path.join(METHOD_COMPARE_DIR, "train_acc_NaiveBayes(Categorical).csv"),
    },
    "SVM": {
        "file": os.path.join(METHOD_COMPARE_DIR, "test_pred_SVM.csv"),
        "train_acc": os.path.join(METHOD_COMPARE_DIR, "train_acc_SVM.csv"),
    },
}


# =====================================================================
#  数据加载
# =====================================================================


def _find_latest_cep_log():
    """找到最新的 CEP leave-one 日志文件。"""
    pattern = os.path.join(LOGS_DIR, "cep_leave_one_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 CEP 日志：{pattern}")
    return files[-1]


def _parse_dict_pred(dict_str):
    """解析 '{1: 0, 2: 1, ...}' 字符串 → 多数投票标签。"""
    d = ast.literal_eval(dict_str)
    values = list(d.values())
    return int(np.bincount(values).argmax())


def _find_best_feature_num(test_pred_path):
    """
    从 test_pred 文件中找到留一验证准确率最高的特征数。
    如果有多个特征数准确率相同，选择特征数最大的。

    Returns
    -------
    best_k : int
    test_pred_df : pd.DataFrame — 用于 per-model 热图
    """
    test_df = pd.read_csv(test_pred_path, index_col=0)
    labels = test_df["label"].astype(int)

    # 特征列：数字列名
    feat_cols = [c for c in test_df.columns if c.isdigit() or (
        isinstance(c, str) and c.lstrip('-').isdigit())]
    feat_cols_sorted = sorted(feat_cols, key=lambda x: int(x))

    # 计算每个特征数的留一验证准确率
    test_acc = {}
    for fc in feat_cols_sorted:
        preds = test_df[fc].astype(int)
        test_acc[fc] = (preds == labels).mean()

    # 找最高准确率，tie-break 用更大的特征数
    best_acc = max(test_acc.values())
    best_k = max(int(fc) for fc, acc in test_acc.items() if acc == best_acc)
    return best_k, test_df


def load_method(method_name, config):
    """加载单个方法的预测结果 → DataFrame(species, label, pred)。

    对于四模型（有 train_acc 配置），使用最优特征数的预测。
    """
    fpath = config["file"]
    if not os.path.exists(fpath):
        print(f"  [WARN] 文件不存在，跳过 {method_name}: {fpath}")
        return None

    # ====== 四模型：使用最优特征数 ======
    if "train_acc" in config:
        best_k, test_df = _find_best_feature_num(fpath)
        print(f"    最优特征数: {best_k}")
        preds = test_df[str(best_k)].astype(int)
        labels = test_df["label"].astype(int) if "label" in test_df.columns else pd.Series(
            np.nan, index=test_df.index)
        return pd.DataFrame({"label": labels, "pred": preds}, index=test_df.index)

    df = pd.read_csv(fpath)
    if "species" in df.columns:
        df = df.set_index("species")

    label_col = config.get("label_col", "label")
    pred_col = config["pred_col"]

    if config.get("is_dict"):
        preds = df[pred_col].apply(_parse_dict_pred)
    else:
        preds = df[pred_col].astype(int)

    labels = df[label_col].astype(int) if label_col in df.columns else pd.Series(
        np.nan, index=df.index
    )
    return pd.DataFrame({"label": labels, "pred": preds}, index=df.index)


def load_cep():
    """加载完整 CEP 流水线结果。"""
    log_path = _find_latest_cep_log()
    print(f"  CEP 日志: {os.path.basename(log_path)}")
    df = pd.read_csv(log_path, index_col=0)
    return pd.DataFrame({
        "label": df["true_label"].astype(int),
        "pred": df["pred_label"].astype(int),
    }, index=df.index)


# =====================================================================
#  指标计算
# =====================================================================


def compute_metrics(y_true, y_pred):
    """计算二分类指标。"""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def build_error_matrix(results, label_col="label", pred_col="pred"):
    """
    汇总所有方法 → 矩阵 (method × species)，其中：
      0 = 预测错误, 1 = 预测正确
    """
    all_species = set()
    for df in results.values():
        if df is not None:
            all_species.update(df.index.tolist())
    all_species = sorted(all_species)

    mat = {}
    for method_name, df in results.items():
        if df is None:
            continue
        row = {}
        for sp in all_species:
            if sp in df.index:
                row[sp] = int(df.loc[sp, label_col] == df.loc[sp, pred_col])
            else:
                row[sp] = np.nan
        mat[method_name] = row
    return pd.DataFrame(mat, index=all_species).T


def compute_metrics_matrix(results):
    """计算每个方法的指标。"""
    metrics = {}
    for method_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        y_true = df["label"].values
        y_pred = df["pred"].values
        metrics[method_name] = compute_metrics(y_true, y_pred)
    return pd.DataFrame(metrics)


# =====================================================================
#  可视化
# =====================================================================


def _load_metadata():
    """加载元数据：物种名 → 中文名, order_chinese_new, label 映射。"""
    meta = pd.read_csv(METADATA_1_CSV, index_col=0)
    species_cn = meta["species_chinese"].to_dict()
    species_order = meta["order_chinese_new"].to_dict()
    species_label = meta["label"].to_dict()
    # 特殊处理几个命名不一致的
    species_cn["Typhlomys_daloushanensis"] = "猪尾鼠"
    species_cn["Balaenoptera_acutorostrata"] = "小须鲸"
    species_cn["Shrew_mole"] = "鼩鼹"
    species_cn["shrew_mole"] = "鼩鼹"
    species_order["Typhlomys_daloushanensis"] = species_order.get("ZWS", "啮齿目")
    species_order["Balaenoptera_acutorostrata"] = species_order.get(
        "Balaenoptera_acutorostrata_scammoni", "鲸目")
    species_order["Shrew_mole"] = species_order.get("shrew_mole", "真盲缺目")
    return species_cn, species_order, species_label


def _sort_species_by_order(species_list, species_order, species_label):
    """按 order_chinese_new 分组排列物种列表。

    排序规则：先按 order 分组，同 order 内按 label（回声=1 在后）。

    Returns
    -------
    sorted_species : list — 按目排序后的物种名列表
    order_boundaries : list of (order_name, start_idx, end_idx)
    """
    # 构建 (order, label, species_name) 三元组
    triples = []
    for sp in species_list:
        order = species_order.get(sp, "未知")
        lbl = int(species_label.get(sp, 0))
        triples.append((order, lbl, sp))
    # 排序：先 order，再 label（非回声=0 在前，回声=1 在后）
    triples.sort(key=lambda x: (x[0], x[1], x[2]))

    sorted_species = [t[2] for t in triples]

    # 找出 order 边界
    order_boundaries = []
    if triples:
        prev_order = triples[0][0]
        start = 0
        for i, t in enumerate(triples):
            if t[0] != prev_order:
                order_boundaries.append((prev_order, start, i))
                prev_order = t[0]
                start = i
        order_boundaries.append((prev_order, start, len(triples)))

    return sorted_species, order_boundaries


def make_feature_count_heatmap(test_pred_df, true_labels_dict, model_name,
                                best_k, out_path, figsize=(18, 8)):
    """
    绘制单模型不同特征数的 error 热图 + 指标热图。

    行=该模型在1~30特征数下的 error 并集（按 order 排序），列=特征数 1~30。
    上半部分：error 热图（0=错误, 1=正确）
    下半部分：指标热图（Accuracy/Precision/Recall/F1 × 特征数）
    """
    sns.set_theme(style="white")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    species_cn, species_order_map, species_label_map = _load_metadata()

    # ---- 特征列 ----
    feat_cols = sorted(
        [c for c in test_pred_df.columns if c.isdigit() or (
            isinstance(c, str) and c.lstrip('-').isdigit())],
        key=lambda x: int(x))

    # ---- 计算该模型自身的 error 并集 ----
    labels = test_pred_df["label"].astype(int)
    all_species = test_pred_df.index.tolist()
    own_error_species = set()
    for fc in feat_cols:
        preds = test_pred_df[fc].astype(int)
        for sp in all_species:
            if preds.loc[sp] != labels.loc[sp]:
                own_error_species.add(sp)

    # 按 order 排序
    own_error_sorted, _ = _sort_species_by_order(
        list(own_error_species), species_order_map, species_label_map)

    # ---- 构建 error 矩阵：本模型 error 并集 × 特征数 ----
    show_species = [s for s in own_error_sorted if s in test_pred_df.index]
    error_data = {}
    for fc in feat_cols:
        row = {}
        preds = test_pred_df[fc].astype(int)
        for sp in show_species:
            row[sp] = int(preds.loc[sp] == labels.loc[sp])
        error_data[fc] = row
    error_mat = pd.DataFrame(error_data, index=show_species).T

    # 重命名物种为中文
    rename_map = {sp: species_cn.get(sp, sp) for sp in error_mat.columns}
    error_mat = error_mat.rename(columns=rename_map)

    # ---- 构建指标矩阵 ----
    metrics_rows = {}
    for fc in feat_cols:
        preds = test_pred_df[fc].astype(int)
        labels = test_pred_df["label"].astype(int)
        metrics_rows[fc] = {
            "Accuracy": accuracy_score(labels, preds),
            "Precision": precision_score(labels, preds, zero_division=0),
            "Recall": recall_score(labels, preds, zero_division=0),
            "F1": f1_score(labels, preds, zero_division=0),
        }
    metrics_df = pd.DataFrame(metrics_rows)  # index=metrics, columns=feat_num

    # ---- 顶部标签条（按 order 排序后的物种 label）----
    cn_to_orig = {v: k for k, v in rename_map.items()}
    ordered_labels = [
        int(true_labels_dict.get(cn_to_orig.get(cn, cn), 0))
        for cn in error_mat.columns
    ]

    n_species = error_mat.shape[1]

    fig = plt.figure(figsize=figsize)
    # Grid: 3 rows — label bar, error heatmap, metrics heatmap
    # 增大 hspace 防止物种名被下方指标热图遮盖
    gs = plt.GridSpec(3, 1, height_ratios=[0.3, n_species * 0.25 + 0.5, 1.8],
                      hspace=0.35)

    # --- A: 顶部标签条 ---
    ax_label = fig.add_subplot(gs[0])
    cmap_label = ListedColormap(["#377eb8", "#e41a1c"])
    sns.heatmap([ordered_labels], ax=ax_label, cbar=False, cmap=cmap_label,
                xticklabels=False, yticklabels=["Label"])
    ax_label.set_yticks([0.5])
    ax_label.set_yticklabels(["Echolocation"], rotation=0, fontsize=9,
                             fontweight="bold")

    # --- B: Error 热图（方法→特征数 × 物种）---
    ax_err = fig.add_subplot(gs[1])
    cmap_err = ListedColormap(["#4d4d4d", "#f0f0f0"])
    sns.heatmap(error_mat, ax=ax_err, cbar=False, cmap=cmap_err,
                linewidths=0.3, linecolor="white",
                xticklabels=True, yticklabels=True)
    ax_err.set_xticklabels(ax_err.get_xticklabels(), rotation=45, ha="right",
                           fontsize=7)
    # 最优特征数用红色字体标注（y轴）
    best_k_str = str(best_k)
    for label in ax_err.get_yticklabels():
        if label.get_text() == best_k_str:
            label.set_color("#e41a1c")
            label.set_fontweight("bold")
            label.set_fontsize(9)

    # --- C: 指标热图 ---
    ax_met = fig.add_subplot(gs[2])
    sns.heatmap(metrics_df, ax=ax_met, annot=True, fmt=".2f", cmap="viridis",
                cbar=False, xticklabels=True, yticklabels=True,
                annot_kws={"size": 6})
    ax_met.set_xticklabels(ax_met.get_xticklabels(), rotation=0, fontsize=7)
    ax_met.tick_params(axis="y", labelsize=8)
    # 最优特征数用红色字体标注（x轴）
    for label in ax_met.get_xticklabels():
        if label.get_text() == best_k_str:
            label.set_color("#e41a1c")
            label.set_fontweight("bold")
            label.set_fontsize(9)

    # ---- 图例 ----
    legend_elements = [
        Patch(facecolor="#e41a1c", label="Echolocation"),
        Patch(facecolor="#377eb8", label="Non-Echolocation"),
        Patch(facecolor="#4d4d4d", label="Wrong prediction"),
        Patch(facecolor="#f0f0f0", label="Correct prediction"),
        Patch(edgecolor="#e41a1c", facecolor="none", linewidth=2,
              label=f"Best k={best_k}"),
    ]
    ax_err.legend(handles=legend_elements, bbox_to_anchor=(0, -0.55),
                  loc="upper left", ncol=5, frameon=True, prop={"size": 7})

    plt.suptitle(f"{model_name} — Error & Metrics across Feature Counts",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  图表已保存: {out_path}")


def make_heatmap(error_matrix, metrics_df, true_labels_dict, title, out_path,
                 show_species=None, figsize=(14, 6)):
    """
    绘制方法 × 物种 的预测错误热图 + 右侧指标。

    Parameters
    ----------
    error_matrix : pd.DataFrame — method × species, 0=错误 1=正确
    metrics_df : pd.DataFrame — index=metrics, columns=methods
    true_labels_dict : dict — {species_name: 0/1}
    show_species : list or None — 要展示的物种列表（已按 order 排序）
    """
    sns.set_theme(style="white")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    species_cn, species_order, species_label = _load_metadata()

    # 筛选要展示的物种
    mat = error_matrix.copy()
    if show_species is not None:
        mat = mat.loc[:, [s for s in show_species if s in mat.columns]]

    # 构建 original_name → 中文名 映射
    rename_map = {sp: species_cn.get(sp, sp) for sp in mat.columns}
    mat = mat.rename(columns=rename_map)

    methods = mat.index.tolist()
    n_methods = len(methods)
    n_species = mat.shape[1]

    # 获取展示物种的真实标签（按 mat 列顺序，即 order 排序后的顺序）
    cn_to_orig = {v: k for k, v in rename_map.items()}
    ordered_labels = [true_labels_dict.get(cn_to_orig.get(cn, cn), 0) for cn in mat.columns]

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, width_ratios=[max(10, n_species), 2],
                      height_ratios=[0.4, n_methods * 0.4 + 1],
                      hspace=0.05, wspace=0.05)

    # ---- 顶部标签条 ----
    ax_label = fig.add_subplot(gs[0, 0])
    cmap_label = ListedColormap(["#377eb8", "#e41a1c"])
    sns.heatmap([ordered_labels], ax=ax_label, cbar=False, cmap=cmap_label,
                xticklabels=False, yticklabels=["Echolocation"])
    ax_label.set_yticks([0.5])
    ax_label.set_yticklabels(["Echolocation"], rotation=0, fontsize=10,
                             fontweight="bold")

    # ---- 主热图 ----
    ax_main = fig.add_subplot(gs[1, 0])
    cmap_main = ListedColormap(["#4d4d4d", "#f0f0f0"])
    sns.heatmap(mat, ax=ax_main, cbar=False, cmap=cmap_main,
                linewidths=0.5, linecolor="white",
                xticklabels=True, yticklabels=True)
    ax_main.set_xticklabels(
        ax_main.get_xticklabels(), rotation=45, ha="right", fontsize=8
    )
    ax_main.tick_params(axis="y", labelsize=10)

    # ---- 右侧指标 ----
    ax_metrics = fig.add_subplot(gs[1, 1])
    plot_metrics = metrics_df.T.loc[
        [m for m in methods if m in metrics_df.columns],
        ["Recall", "F1"]
    ]
    sns.heatmap(plot_metrics, ax=ax_metrics, annot=True, fmt=".2f",
                cmap="viridis", cbar=False, yticklabels=False,
                annot_kws={"size": 8})
    ax_metrics.set_xticks([0.5, 1.5])
    ax_metrics.set_xticklabels(["Recall", "F1"], rotation=45)
    ax_metrics.tick_params(axis="x", labelsize=10)

    # ---- 图例 ----
    legend_elements = [
        Patch(facecolor="#e41a1c", label="Echolocation"),
        Patch(facecolor="#377eb8", label="Non-Echolocation"),
        Patch(facecolor="#4d4d4d", label="Wrong prediction"),
        Patch(facecolor="#f0f0f0", label="Correct prediction"),
    ]
    ax_main.legend(handles=legend_elements, bbox_to_anchor=(0, -0.35),
                   loc="upper left", ncol=4, frameon=True, prop={"size": 8})

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  图表已保存: {out_path}")


# =====================================================================
#  main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="多方法评估汇总")
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                        help=f"输出目录（默认: {RESULTS_DIR}）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================================================================
    #  1. 加载所有方法
    # ==================================================================
    print("=" * 60)
    print("  加载方法结果")
    print("=" * 60)

    results = {}

    # CEP 完整流水线
    print("\n  [loading] CEP ...")
    try:
        results["CEP"] = load_cep()
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")

    # ESL & ESL-PSC & 消融 & 方法对比
    for method_name, config in METHOD_CONFIG.items():
        print(f"  [loading] {method_name} ...")
        df = load_method(method_name, config)
        if df is not None:
            results[method_name] = df

    print(f"\n  共加载 {len(results)} 个方法")

    # ==================================================================
    #  2. 计算指标
    # ==================================================================
    print("\n" + "=" * 60)
    print("  各方法指标")
    print("=" * 60)

    metrics_df = compute_metrics_matrix(results)
    print(metrics_df.round(4).to_string())

    # ==================================================================
    #  3. 生成错误矩阵
    # ==================================================================
    error_all = build_error_matrix(results)

    # 找出有错误的物种（至少被1个方法误判）
    error_species = []
    for sp in error_all.columns:
        if (error_all[sp] == 0).any():
            error_species.append(sp)

    # 按 order_chinese_new 排序 error_species
    _, species_order_map, species_label_map = _load_metadata()
    error_species_sorted, order_boundaries = _sort_species_by_order(
        error_species, species_order_map, species_label_map)

    print(f"\n  共 {len(error_species_sorted)} 个物种存在至少1次误判")
    print(f"  按目分组: {', '.join([f'{n}({e-s})' for n, s, e in order_boundaries])}")

    # ==================================================================
    #  4. 保存汇总 CSV
    # ==================================================================
    # 保存完整预测表
    pred_table = {}
    for method_name, df in results.items():
        if df is None:
            continue
        pred_table[method_name] = df["pred"]
    pred_df = pd.DataFrame(pred_table)
    pred_df["true_label"] = list(results.values())[0]["label"].reindex(
        pred_df.index
    )
    pred_csv = os.path.join(args.output_dir, "all_methods_pred.csv")
    pred_df.to_csv(pred_csv)
    print(f"\n  完整预测表: {pred_csv}")

    # 错误详情
    error_detail = {}
    for method_name in results:
        if method_name not in error_all.index:
            continue
        row = error_all.loc[method_name]
        wrong = row[row == 0].index.tolist()
        error_detail[method_name] = wrong

    # ==================================================================
    #  5. 可视化
    # ==================================================================
    print("\n" + "=" * 60)
    print("  生成可视化")
    print("=" * 60)

    # 获取真实标签字典（从 CEP log）
    log_path = _find_latest_cep_log()
    cep_df = pd.read_csv(log_path, index_col=0)
    true_labels_dict = cep_df["true_label"].astype(int).to_dict()

    # 图1: 方法对比（CEP + ESL + ESL-PSC + 四模型）— 仅展示 error 物种，按目排序
    compare_methods = ["CEP", "ESL", "ESL-PSC",
                       "Random Forest", "Logistic Regression",
                       "Naive Bayes", "SVM"]
    compare_methods = [m for m in compare_methods if m in error_all.index]
    if compare_methods:
        make_heatmap(
            error_all.loc[compare_methods],
            metrics_df,
            true_labels_dict,
            title="Method Comparison — Prediction Errors",
            out_path=os.path.join(args.output_dir, "compare_eval_plot.svg"),
            show_species=error_species_sorted,
            figsize=(14, 5),
        )

    # 图2: 消融实验 — 仅展示 error 物种，按目排序
    ablation_methods = ["CEP", "CEP (coldstart)", "CEP w/o coldstart",
                        "Global RF", "CEP w/o featsel", "CEP NMI-only"]
    ablation_methods = [m for m in ablation_methods if m in error_all.index]
    if ablation_methods:
        make_heatmap(
            error_all.loc[ablation_methods],
            metrics_df,
            true_labels_dict,
            title="Ablation Study — Prediction Errors",
            out_path=os.path.join(args.output_dir, "ablation_eval_plot.svg"),
            show_species=error_species_sorted,
            figsize=(14, 5),
        )

    # ==================================================================
    #  5.5 四模型 per-model 特征数分析图
    # ==================================================================
    print("\n" + "=" * 60)
    print("  生成四模型特征数分析图")
    print("=" * 60)

    model_display_names = [
        ("Random Forest", "test_pred_RandomForest.csv"),
        ("Logistic Regression", "test_pred_LogisticRegression.csv"),
        ("Naive Bayes", "test_pred_NaiveBayes(Categorical).csv"),
        ("SVM", "test_pred_SVM.csv"),
    ]
    for model_name, pred_file in model_display_names:
        pred_path = os.path.join(METHOD_COMPARE_DIR, pred_file)
        if not os.path.exists(pred_path):
            print(f"  [WARN] 跳过 {model_name}: 文件不存在")
            continue
        best_k, test_df = _find_best_feature_num(pred_path)
        print(f"  [{model_name}] best k={best_k}, 生成特征数分析图...")
        make_feature_count_heatmap(
            test_df, true_labels_dict, model_name,
            best_k,
            out_path=os.path.join(
                args.output_dir,
                f"feature_count_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.svg"
            ),
        )

    # ==================================================================
    #  6. 打印报告
    # ==================================================================
    print("\n" + "=" * 60)
    print("  报告摘要")
    print("=" * 60)

    for method_name in results:
        if method_name not in error_all.index:
            continue
        n_wrong = int((error_all.loc[method_name] == 0).sum())
        n_total = int(error_all.loc[method_name].notna().sum())
        wrong_species = error_detail.get(method_name, [])
        # 显示中文名
        species_cn, _, _ = _load_metadata()
        wrong_cn = [species_cn.get(s, s) for s in wrong_species]
        print(f"\n  {method_name}: {n_wrong}/{n_total} 错误")
        if wrong_cn:
            print(f"    误判物种: {', '.join(wrong_cn[:10])}"
                  + (f" ... (+{len(wrong_cn) - 10})" if len(wrong_cn) > 10 else ""))

    # ==================================================================
    #  6.5 CEP vs CEP (coldstart) 差异分析
    # ==================================================================
    print("\n" + "=" * 60)
    print("  CEP vs CEP (coldstart) 差异分析")
    print("=" * 60)
    print("""
  【完整 CEP (leave_one_eval.py)】
    - 翼手目/鲸目: RF (n_estimators=100, max_features=10)
    - 其余目:     趋同突变计数法 (top_k=500)
    - 特征排序:   cover_score × NMI
    - OrdinalEncoder: 每个 feature_num 单独 fit

  【消融 CEP (coldstart) (ablation_study.py coldstart_score)】
    - 所有物种:   RF (n_estimators=100, max_features=500)
    - 特征排序:   cover_score × NMI (与 CEP 相同)
    - OrdinalEncoder: 一次性 fit 全部特征后切片

  关键差异:
    1. 非蝙蝠/非鲸目物种 (~74个): CEP用趋同突变计数法, 消融用RF
       → 两种算法完全不同, 这些物种的预测会有显著差异
    2. 蝙蝠/鲸目物种 (~30个): 都用RF, 但CEP只取top 10特征,
       消融用最多500特征 → 特征数差异导致概率分布不同
    3. OrdinalEncoder 策略: 每个feature_num单独fit vs 一次性fit
       → 编码器看到的类别范围不同, 可能影响RF预测
""")

    # 保存指标
    metrics_csv = os.path.join(args.output_dir, "all_methods_metrics.csv")
    metrics_df.round(4).to_csv(metrics_csv)
    print(f"\n  指标已保存: {metrics_csv}")

    # 保存错误详情
    error_detail_df = pd.DataFrame(
        {k: pd.Series(v) for k, v in error_detail.items()}
    )
    error_csv = os.path.join(args.output_dir, "all_methods_errors.csv")
    error_detail_df.to_csv(error_csv, index=False)
    print(f"  错误详情: {error_csv}")

    print("\n" + "=" * 60)
    print("  评估完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
