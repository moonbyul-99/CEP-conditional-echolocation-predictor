"""
CEP 留一验证核心模块（Leave-One-Species-Out Evaluation）

实现逻辑严格对齐 notebook/04_方法对比_v2.ipynb 中的 process_species()。

CEP 预测算法（对齐论文 cep_draft_0523.pdf）：
    1. 对每个物种，加载预计算的特征排序数据（103_leave/{species_id}/）
    2. 根据 cover_score × NMI 重新排序特征
    3. 过滤 Eco_Mode == '-' 的特征（非回声趋同位点）
    4. 根据物种所在目构建参考物种列表（ref_list）
    5. 预测策略：
       - 翼手目 / 鲸目：使用 RandomForest（top 10 特征）
       - 其他目：使用趋同突变计数法（eco_mutation count vs ref_max）
    6. 对多个 feature_num 的概率取均值，> 0.5 判定为回声

数据来源：
    data/leave_one/{species_id}/
        df_feature.csv  — 特征矩阵（104 物种 × 10000 特征）
        df_meta.csv     — 元数据（104 物种，含 order_chinese_new, label 等）
        df_summary.csv  — 特征评分汇总（含 NMI, eco_cover, Eco_Mode 等）
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

try:
    from src.config import LEAVE_ONE_DIR, LOGS_DIR, N_CPU
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.config import LEAVE_ONE_DIR, LOGS_DIR, N_CPU


# =========================================================================
#  特征评分参数
# =========================================================================
SCORE_MAP_4 = {4: 1, 3: 0.95, 2: 0.9, 1: 0.1, 0: 0}
SCORE_MAP_5 = {5: 1, 4: 0.90, 3: 0.75, 2: 0.5, 1: 0.1, 0: 0}

# RF 参数（翼手目/鲸目专用）
RF_N_ESTIMATORS = 100
RF_MAX_FEATURE_NUM = 10  # 使用 top 10 特征

# 计数法参数（其他目）
COUNT_MIN_FRAC = 0.07   # 低于此比例的特征数，计数置零
COUNT_MAX_SCORE = 10     # score 上限
COUNT_PROB_OFFSET = 0.01 # 概率偏移量


# =========================================================================
#  数据加载
# =========================================================================

def load_species_data(species_id, base_dir=None):
    """
    加载单个物种的预计算数据。

    Parameters
    ----------
    species_id : str
    base_dir : str or None
        103_leave 数据根目录，默认 LEAVE_ONE_DIR

    Returns
    -------
    dict : feature_df, meta_df, summary_df
    """
    if base_dir is None:
        base_dir = LEAVE_ONE_DIR

    dir_path = os.path.join(base_dir, species_id)
    feature_df = pd.read_csv(os.path.join(dir_path, 'df_feature.csv'), index_col=0)
    meta_df    = pd.read_csv(os.path.join(dir_path, 'df_meta.csv'),    index_col=0)
    summary_df = pd.read_csv(os.path.join(dir_path, 'df_summary.csv'), index_col=0)

    assert (meta_df.index == feature_df.index).all()
    assert (summary_df.index == feature_df.columns).all()

    return {
        'feature_df': feature_df,
        'meta_df': meta_df,
        'summary_df': summary_df,
    }


# =========================================================================
#  特征重排序
# =========================================================================

def rescore_features(feature_df, summary_df):
    """
    根据 cover_score × NMI 重新排序特征，并过滤 Eco_Mode == '-' 的位点。

    对齐 notebook 逻辑：
        - 如果 5 列 coverage 中有全 0 列，用 SCORE_MAP_4；否则用 SCORE_MAP_5
        - cover_score = eco_cover.map(score_map)
        - score = cover_score × NMI
        - 过滤 Eco_Mode != '-'

    Returns
    -------
    feature_df : 重排 + 过滤后的特征矩阵
    summary_df : 重排 + 过滤后的评分汇总
    """
    # 判断使用 4 级还是 5 级评分
    if summary_df.iloc[:, :5].sum(axis=0).min() == 0:
        summary_df = summary_df.copy()
        summary_df.loc[:, 'cover_score'] = summary_df['eco_cover'].map(SCORE_MAP_4)
    else:
        summary_df = summary_df.copy()
        summary_df.loc[:, 'cover_score'] = summary_df['eco_cover'].map(SCORE_MAP_5)

    summary_df.loc[:, 'score'] = summary_df['cover_score'] * summary_df['NMI']
    summary_df = summary_df.sort_values(by='score', ascending=False)
    feature_df = feature_df.loc[:, summary_df.index]

    # 过滤 Eco_Mode == '-' 的位点
    idx = summary_df['Eco_Mode'].values != '-'
    feature_df = feature_df.loc[:, idx]
    summary_df = summary_df.loc[idx, :]

    return feature_df, summary_df


# =========================================================================
#  参考物种列表构建
# =========================================================================

def build_ref_list(species_id, meta_df):
    """
    根据物种所在目构建参考物种列表（训练集）。

    对齐 notebook 逻辑：
        - 鲸目: 鲸目 + 偶蹄目（排除自身）
        - 翼手目: 翼手目 + 啮齿目（排除自身）
        - 真盲缺目: 指定 4 个物种
        - 啮齿目: 同目物种（排除自身；非 ZWS 时额外排除 ZWS）
        - 攀鼩目: 指定 5 个物种
        - 兔形目: 指定 6 个物种（排除自身）
        - 其他目（食肉目/奇蹄目/偶蹄目/非洲兽目/灵长目）: 同目且 label==0
    """
    species_order = meta_df.loc[species_id, 'order_chinese_new']

    if species_order == '鲸目':
        tmp = meta_df.loc[meta_df['order_chinese_new'].isin([species_order, '偶蹄目']), :]
        ref_list = tmp.index.to_list()
        if species_id in ref_list:
            ref_list.remove(species_id)

    elif species_order == '翼手目':
        tmp = meta_df.loc[meta_df['order_chinese_new'].isin([species_order, '啮齿目']), :]
        ref_list = tmp.index.to_list()
        if species_id in ref_list:
            ref_list.remove(species_id)

    elif species_order == '真盲缺目':
        ref_list = ['Condylura_cristata', 'Ceratotherium_simum_simum',
                     'Equus_asinus', 'Equus_quagga']
        if species_id in ref_list:
            ref_list.remove(species_id)

    elif species_order == '啮齿目':
        tmp = meta_df.loc[meta_df['order_chinese_new'].values == species_order, :]
        ref_list = tmp.index.to_list()
        if species_id in ref_list:
            ref_list.remove(species_id)
        if species_id != 'ZWS' and 'ZWS' in ref_list:
            ref_list.remove('ZWS')  # 啮齿目中唯一的非回声物种

    elif species_order == '攀鼩目':
        ref_list = ['Propithecus_coquereli', 'Gorilla_gorilla_gorilla',
                     'Pan_paniscus', 'Pan_troglodytes', 'Marmota_monax']

    elif species_order == '兔形目':
        ref_list = ['Propithecus_coquereli', 'Gorilla_gorilla_gorilla',
                     'Pan_paniscus', 'Ochotona_princeps', 'Ochotona_curzoniae']
        if species_id in ref_list:
            ref_list.remove(species_id)

    else:
        # 其他目：食肉目、奇蹄目、偶蹄目、非洲兽目、灵长目
        tmp = meta_df.loc[meta_df['order_chinese_new'].values == species_order, :]
        tmp = tmp.loc[tmp['label'].values == 0, :]
        ref_list = tmp.index.to_list()
        if species_id in ref_list:
            ref_list.remove(species_id)

    return ref_list


# =========================================================================
#  CEP 预测
# =========================================================================

def predict_species(species_id, feature_df, meta_df, summary_df, ref_list,
                    top_k=500):
    """
    CEP 核心预测函数。

    - 翼手目/鲸目：RandomForest（top 10 特征），返回 10 个概率
    - 其他目：趋同突变计数法，返回 top_k 个概率

    对齐 notebook 中 process_species() 的完整逻辑。
    """
    species_order = meta_df.loc[species_id, 'order_chinese_new']

    if species_order in ['翼手目', '鲸目']:
        # ========== RandomForest 预测 ==========
        prob_list = []
        for feature_num in range(1, RF_MAX_FEATURE_NUM + 1):
            train_feature = feature_df.loc[ref_list, :].iloc[:, :feature_num]
            train_label   = meta_df.loc[ref_list, 'label'].values
            target_feature = feature_df.loc[[species_id], :].iloc[:, :feature_num]

            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(train_feature)

            X_train  = encoder.transform(train_feature)
            X_target = encoder.transform(target_feature)

            # RF 处理：将 -1（未知类别）替换为 0
            X_train_rf  = np.where(X_train == -1, 0, X_train)
            X_target_rf = np.where(X_target == -1, 0, X_target)

            model = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=1
            )
            model.fit(X_train_rf, train_label)
            prob_pred = model.predict_proba(X_target_rf)[0, 1]
            prob_list.append(prob_pred)

    else:
        # ========== 趋同突变计数法预测 ==========
        eco_mode_values = summary_df['Eco_Mode'].values
        prob_list = []

        for feature_num in range(1, top_k + 1):
            all_feature = feature_df.iloc[:, :feature_num]
            eco_mu = eco_mode_values[:feature_num]

            # 预测物种的趋同突变计数
            pred_count = (all_feature.loc[species_id].values == eco_mu).sum()
            # 参考物种的趋同突变计数（取最大值）
            ref_count = (all_feature.loc[ref_list, :].values == eco_mu).sum(axis=1)
            ref_max = ref_count.max()

            # 过低计数置零
            if pred_count < int(COUNT_MIN_FRAC * feature_num):
                pred_count = 0

            score = (COUNT_PROB_OFFSET + pred_count) / (COUNT_PROB_OFFSET + ref_max)
            score = min(score, COUNT_MAX_SCORE)
            prob = score / (1 + score) - COUNT_PROB_OFFSET
            prob_list.append(prob)

    return prob_list


# =========================================================================
#  单物种评估任务（供多进程调用）
# =========================================================================

def process_one_species(species_id, base_dir=None, top_k=500):
    """
    单个物种的完整 CEP 留一评估流程。

    Returns
    -------
    (species_id, prob_list, pred_label, true_label, error)
    """
    try:
        data = load_species_data(species_id, base_dir=base_dir)
        feature_df, meta_df, summary_df = (
            data['feature_df'], data['meta_df'], data['summary_df']
        )

        # 过滤 label==2 的物种（不参与评估的未知标签物种）
        valid_idx = meta_df['label'].values != 2
        meta_df = meta_df.loc[valid_idx, :]
        feature_df = feature_df.loc[valid_idx, :]

        # 特征重排序 + Eco_Mode 过滤
        feature_df, summary_df = rescore_features(feature_df, summary_df)

        # 构建参考物种列表
        ref_list = build_ref_list(species_id, meta_df)

        # CEP 预测
        prob_list = predict_species(
            species_id, feature_df, meta_df, summary_df, ref_list, top_k=top_k
        )

        true_label = int(meta_df.loc[species_id, 'label'])
        mean_prob = np.mean(prob_list)
        pred_label = int(mean_prob > 0.5)

        return (species_id, prob_list, pred_label, true_label, None)

    except Exception as e:
        return (species_id, None, None, None, str(e))


# =========================================================================
#  多进程批量评估
# =========================================================================

def parallel_processing(base_dir=None, species_list=None, top_k=500,
                        n_cpu=None):
    """
    对全部 104 个物种执行 CEP 留一验证。

    Parameters
    ----------
    base_dir : str or None
        103_leave 数据根目录。默认 LEAVE_ONE_DIR。
    species_list : list or None
        待评估物种列表。默认从 base_dir 中自动获取。
    top_k : int
        计数法扫描的最大特征数（默认 500）。
    n_cpu : int or None
        并行进程数。默认 N_CPU。

    Returns
    -------
    results_df : pd.DataFrame
        包含 species, true_label, pred_label, mean_prob, correct 列
    """
    if base_dir is None:
        base_dir = LEAVE_ONE_DIR
    if n_cpu is None:
        n_cpu = N_CPU

    # 获取物种列表
    if species_list is None:
        # 从任一物种目录的 df_meta.csv 中读取全部有效物种
        sample_meta = pd.read_csv(
            os.path.join(base_dir, os.listdir(base_dir)[0], 'df_meta.csv'),
            index_col=0
        )
        species_list = sample_meta.loc[sample_meta['label'].values != 2].index.to_list()

    print(f"CEP 留一验证：{len(species_list)} 个物种，{n_cpu} 个 CPU")
    print(f"数据目录：{base_dir}")

    task_func = partial(process_one_species, base_dir=base_dir, top_k=top_k)

    results = []
    with Pool(processes=min(n_cpu, len(species_list))) as pool:
        for res in pool.imap_unordered(task_func, species_list):
            results.append(res)
            done = len(results)
            if done % 10 == 0 or done == len(species_list):
                print(f"  [{done}/{len(species_list)}]")

    # 整理结果
    rows = []
    errors = []
    for species_id, prob_list, pred_label, true_label, error in results:
        if error is not None:
            errors.append((species_id, error))
            continue
        mean_prob = float(np.mean(prob_list))
        rows.append({
            'species': species_id,
            'true_label': true_label,
            'pred_label': pred_label,
            'mean_prob': mean_prob,
            'correct': int(pred_label == true_label),
        })

    results_df = pd.DataFrame(rows)
    if len(results_df) > 0:
        results_df = results_df.set_index('species')

    # 汇总
    if len(results_df) > 0:
        acc = results_df['correct'].mean()
        n_correct = results_df['correct'].sum()
        n_total = len(results_df)
        print(f"\nCEP 准确率: {n_correct}/{n_total} = {acc:.4f}")

    if errors:
        print(f"\n{len(errors)} 个物种评估失败:")
        for sp, err in errors:
            print(f"  {sp}: {err}")

    # 保存结果
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOGS_DIR, f"cep_leave_one_{timestamp}.csv")
    results_df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"结果已保存：{csv_path}")

    return results_df
