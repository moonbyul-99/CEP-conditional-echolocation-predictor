#!/usr/bin/env python3
"""
CEP 消融实验脚本

运行 5 种 RF 变体，对比不同策略对预测准确率的影响：
  1. local_no_coldstart  - 局部 RF（同目训练集），无冷启动
  2. global_rf              - 全局 RF（全部物种训练集）
  3. coldstart_score  - **CEP 的 RF 组件**（冷启动 + cover_score×NMI 排序）
                        即 leave_one_eval.py 中翼手目/鲸目使用的完整 RF 策略，
                        此处对所有 104 物种统一应用以衡量其泛化能力
  4. coldstart_random - 冷启动 + 随机特征（对照：检验特征排序的贡献）
  5. coldstart_nmi    - 冷启动 + 仅 NMI 排序（对照：检验 cover_score 的贡献）

等价于 notebook/04_方法对比_v3.ipynb

注意：CEP 完整方法（leave_one_run.py）对翼手目/鲸目使用 coldstart_score (RF)，
      对其他目使用趋同突变计数法。本脚本的 coldstart_score 变体将 RF 策略
      扩展到所有物种，用于消融对比。

用法：
    cd CEP_project
    python scripts/ablation_study.py --n-cpu 64
"""

import argparse
import os
import sys
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import LEAVE_ONE_DIR, RESULTS_DIR, N_CPU

SCORE_MAP_4 = {4: 1, 3: 0.95, 2: 0.9, 1: 0.1, 0: 0}
SCORE_MAP_5 = {5: 1, 4: 0.90, 3: 0.75, 2: 0.5, 1: 0.1, 0: 0}
RF_N_ESTIMATORS = 100


# =========================================================================
#  辅助函数
# =========================================================================

def _encode_features(feature_df, max_feat):
    """OrdinalEncoder 编码前 max_feat 列特征"""
    raw = feature_df.iloc[:, :max_feat]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X = enc.fit_transform(raw)
    X = np.where(X == -1, 0, X)
    return pd.DataFrame(X, index=feature_df.index)


def _rescore(summary_df):
    """重新计算 cover_score × NMI 排序"""
    s = summary_df.copy()
    if s.iloc[:, :5].sum(axis=0).min() == 0:
        s['cover_score'] = s['eco_cover'].map(SCORE_MAP_4)
    else:
        s['cover_score'] = s['eco_cover'].map(SCORE_MAP_5)
    s['score'] = s['cover_score'] * s['NMI']
    return s.sort_values(by='score', ascending=False)


def _build_ref_list(species_id, species_order, meta_df):
    """构建冷启动参考物种列表（对齐 leave_one_eval.py）"""
    if species_order == '鲸目':
        tmp = meta_df.loc[meta_df['order_chinese_new'].isin([species_order, '偶蹄目']), :]
        ref_list = [s for s in tmp.index if s != species_id]
    elif species_order == '翼手目':
        tmp = meta_df.loc[meta_df['order_chinese_new'].isin([species_order, '啮齿目']), :]
        ref_list = [s for s in tmp.index if s != species_id]
    elif species_order == '真盲缺目':
        ref_list = ['Condylura_cristata', 'Ceratotherium_simum_simum',
                     'Equus_asinus', 'Equus_quagga']
        ref_list = [s for s in ref_list if s != species_id]
    elif species_order == '啮齿目':
        tmp = meta_df.loc[meta_df['order_chinese_new'] == species_order, :]
        ref_list = [s for s in tmp.index if s != species_id]
        if species_id != 'ZWS' and 'ZWS' in ref_list:
            ref_list.remove('ZWS')
    elif species_order == '攀鼩目':
        ref_list = ['Propithecus_coquereli', 'Gorilla_gorilla_gorilla',
                     'Pan_paniscus', 'Pan_troglodytes', 'Marmota_monax']
    elif species_order == '兔形目':
        ref_list = ['Propithecus_coquereli', 'Gorilla_gorilla_gorilla',
                     'Pan_paniscus', 'Ochotona_princeps', 'Ochotona_curzoniae']
        ref_list = [s for s in ref_list if s != species_id]
    else:
        tmp = meta_df.loc[meta_df['order_chinese_new'] == species_order, :]
        tmp = tmp.loc[tmp['label'] == 0, :]
        ref_list = [s for s in tmp.index if s != species_id]
    return ref_list


# =========================================================================
#  5 种消融变体
# =========================================================================

def _run_rf(feature_df, meta_df, train_species, species_id, max_feat):
    """通用 RF 训练 + 预测，返回 {feat_num: pred} 和 {feat_num: train_acc}"""
    X = _encode_features(feature_df, max_feat)
    y = meta_df['label']
    preds, accs = {}, {}
    for fn in range(1, max_feat + 1):
        Xtr = X.loc[train_species, :fn]
        ytr = y.loc[train_species]
        model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                       random_state=42, n_jobs=1)
        model.fit(Xtr, ytr)
        preds[fn] = int(model.predict(X.loc[[species_id], :fn])[0])
        accs[fn] = float(accuracy_score(ytr, model.predict(Xtr)))
    return preds, accs


def variant_local_no_coldstart(species_id, feature_df, meta_df, summary_df, max_feat=30):
    """变体 1：局部 RF，无冷启动（对照：检验冷启动的贡献）
    
    鲸/翼/啮→同目训练集，其他→全局训练集。
    不使用 _build_ref_list() 冷启动策略，直接使用特征 1-30。
    """
    order = meta_df.loc[species_id, 'order_chinese_new']
    if order in ['鲸目', '翼手目', '啮齿目']:
        train = [s for s in meta_df.index
                 if meta_df.loc[s, 'order_chinese_new'] == order and s != species_id]
    else:
        train = [s for s in meta_df.index if s != species_id]
    return _run_rf(feature_df, meta_df, train, species_id, max_feat)


def variant_global_rf(species_id, feature_df, meta_df, summary_df, max_feat=30):
    """变体 2：全局 RF（对照：检验局部训练集 vs 全局训练集的差异）
    
    所有物种都使用全部物种作为训练集（排除自身）。
    """
    train = [s for s in meta_df.index if s != species_id]
    return _run_rf(feature_df, meta_df, train, species_id, max_feat)


def _run_rf_with_probs(feature_df, meta_df, summary_df, species_id,
                        train_species, max_feat=500):
    """RF 概率预测（用于变体 3-5），返回 prob_list
    
    优化：只对最大特征数做一次 OrdinalEncoder，后续切片复用。
    """
    # Eco_Mode 过滤
    idx = summary_df['Eco_Mode'].values != '-'
    fdf = feature_df.loc[:, idx]
    sdf = summary_df.loc[idx, :]

    actual_max = min(max_feat, fdf.shape[1])

    # 一次性编码全部特征
    raw = fdf.iloc[:, :actual_max]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_full = enc.fit_transform(raw)
    X_full = np.where(X_full == -1, 0, X_full)
    X_full_df = pd.DataFrame(X_full, index=fdf.index)

    prob_list = []
    for fn in range(1, actual_max + 1):
        Xtr = X_full_df.loc[train_species, :fn].values
        ytr = meta_df.loc[train_species, 'label'].values
        Xtest = X_full_df.loc[[species_id], :fn].values

        model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                       random_state=42, n_jobs=1)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xtest)
        # 如果训练集只有一类，predict_proba 返回 (n,1) 而非 (n,2)
        if proba.shape[1] == 1:
            prob = 1.0 if model.classes_[0] == 1 else 0.0
        else:
            prob = proba[0, 1]
        prob_list.append(prob)
    return prob_list


def variant_coldstart_score(species_id, feature_df, meta_df, summary_df, max_feat=500):
    """变体 3：**CEP 的 RF 组件**（冷启动 + cover_score×NMI）
    
    与 leave_one_eval.py 中翼手目/鲸目的 predict_species() 完全一致：
      - 使用 _build_ref_list() 构建冷启动参考物种列表
      - 特征按 cover_score × NMI 降序排列
      - 过滤 Eco_Mode == '-' 的位点
      - RF 逐步增加特征数，概率取均值，> 0.5 判定为回声
    区别：CEP 完整方法仅对翼手目/鲸目使用此策略，
         此处扩展到所有 104 物种以进行消融对比。
    """
    sdf = _rescore(summary_df)
    fdf = feature_df.loc[:, sdf.index]
    order = meta_df.loc[species_id, 'order_chinese_new']
    train = _build_ref_list(species_id, order, meta_df)
    prob_list = _run_rf_with_probs(fdf, meta_df, sdf, species_id, train, max_feat)
    mean_prob = np.mean(prob_list)
    return int(mean_prob > 0.5), mean_prob


def variant_coldstart_random(species_id, feature_df, meta_df, summary_df, max_feat=500):
    """变体 4：冷启动 + 随机特征（对照：检验 cover_score×NMI 排序的贡献）
    
    与 coldstart_score 相同的冷启动和 RF 策略，
    但特征顺序随机打乱而非按 score 降序。
    """
    sdf = _rescore(summary_df)
    N = sdf.shape[0]
    np.random.seed(42)
    rand_idx = np.random.choice(N, min(5000, N), replace=False)
    sdf = sdf.iloc[rand_idx, :]
    fdf = feature_df.loc[:, sdf.index]
    order = meta_df.loc[species_id, 'order_chinese_new']
    train = _build_ref_list(species_id, order, meta_df)
    prob_list = _run_rf_with_probs(fdf, meta_df, sdf, species_id, train, max_feat)
    mean_prob = np.mean(prob_list)
    return int(mean_prob > 0.5), mean_prob


def variant_coldstart_nmi(species_id, feature_df, meta_df, summary_df, max_feat=500):
    """变体 5：冷启动 + 仅 NMI 排序（对照：检验 cover_score 的贡献）
    
    与 coldstart_score 相同的冷启动和 RF 策略，
    但特征按 NMI 降序而非 cover_score×NMI 降序。
    """
    sdf = summary_df.copy().sort_values(by='NMI', ascending=False)
    fdf = feature_df.loc[:, sdf.index]
    order = meta_df.loc[species_id, 'order_chinese_new']
    train = _build_ref_list(species_id, order, meta_df)
    prob_list = _run_rf_with_probs(fdf, meta_df, sdf, species_id, train, max_feat)
    mean_prob = np.mean(prob_list)
    return int(mean_prob > 0.5), mean_prob


# =========================================================================
#  主流程
# =========================================================================

def process_one(species_id, base_dir, variants_to_run):
    """加载数据，运行所有变体"""
    try:
        dir_path = os.path.join(base_dir, species_id)
        feature_df = pd.read_csv(os.path.join(dir_path, 'df_feature.csv'), index_col=0)
        meta_df = pd.read_csv(os.path.join(dir_path, 'df_meta.csv'), index_col=0)
        summary_df = pd.read_csv(os.path.join(dir_path, 'df_summary.csv'), index_col=0)

        valid = meta_df['label'].values != 2
        meta_df = meta_df.loc[valid, :]
        feature_df = feature_df.loc[valid, :]

        true_label = int(meta_df.loc[species_id, 'label'])
        results = {'species': species_id, 'true_label': true_label}

        for vname in variants_to_run:
            fn = VARIANTS[vname]
            res = fn(species_id, feature_df, meta_df, summary_df)
            if isinstance(res, tuple) and len(res) == 2:
                pred, prob = res
                results[f'{vname}_pred'] = pred
                results[f'{vname}_prob'] = prob
            else:
                preds, accs = res
                # 取最优准确率对应的预测
                best_fn = max(accs, key=accs.get)
                results[f'{vname}_pred'] = preds[best_fn]
                results[f'{vname}_best_fn'] = best_fn
                results[f'{vname}_best_acc'] = accs[best_fn]
                # 保存所有特征数的预测
                for fn_k, fn_v in preds.items():
                    results[f'{vname}_fn{fn_k}'] = fn_v

        return results

    except Exception as e:
        return {'species': species_id, 'error': str(e)}


VARIANTS = {
    'local_no_cold':    variant_local_no_coldstart,
    'global_rf':        variant_global_rf,
    # 'coldstart_score':  variant_coldstart_score,
    'coldstart_random': variant_coldstart_random,
    'coldstart_nmi':    variant_coldstart_nmi,
}


def main():
    parser = argparse.ArgumentParser(description='CEP 消融实验（5 种 RF 变体）')
    parser.add_argument('--base-dir', type=str, default=LEAVE_ONE_DIR,
                        help=f'103_leave 数据目录（默认：{LEAVE_ONE_DIR}）')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()),
                        help=f'要运行的变体（默认全部：{list(VARIANTS.keys())}）')
    parser.add_argument('--n-cpu', type=int, default=N_CPU,
                        help=f'并行进程数（默认：{N_CPU}）')
    args = parser.parse_args()

    species_list = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])
    print(f"消融实验：{len(species_list)} 个物种，变体 {args.variants}")

    task_func = partial(process_one, base_dir=args.base_dir,
                        variants_to_run=args.variants)

    results = []
    with Pool(processes=min(args.n_cpu, len(species_list))) as pool:
        for res in tqdm(pool.imap_unordered(task_func, species_list),
                        total=len(species_list), desc="Ablation"):
            results.append(res)

    df = pd.DataFrame(results).set_index('species')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ablation_study.csv')
    df.to_csv(out_path)
    print(f"\n结果已保存：{out_path}")

    # 汇总准确率（对有 pred 列的变体）
    for vname in args.variants:
        pred_col = f'{vname}_pred'
        if pred_col in df.columns:
            try:
                valid_df = df.dropna(subset=[pred_col, 'true_label'])
                if len(valid_df) > 0:
                    preds = valid_df[pred_col].apply(lambda x: int(x) if not isinstance(x, dict) else None)
                    labels = valid_df['true_label'].apply(lambda x: int(x))
                    mask = preds.notna()
                    if mask.sum() > 0:
                        acc = (preds[mask].astype(int) == labels[mask].astype(int)).mean()
                        n = (preds[mask].astype(int) == labels[mask].astype(int)).sum()
                        print(f"  {vname}: {n}/{mask.sum()} = {acc:.4f}")
            except Exception as e:
                print(f"  {vname}: 汇总失败 - {e}")

    errors = df[df['error'].notna()] if 'error' in df.columns else pd.DataFrame()
    if len(errors) > 0:
        print(f"\n{len(errors)} 个物种评估失败:")
        for sp, row in errors.iterrows():
            print(f"  {sp}: {row['error']}")


if __name__ == '__main__':
    main()
