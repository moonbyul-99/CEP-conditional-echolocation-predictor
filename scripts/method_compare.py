#!/usr/bin/env python3
"""
多模型留一验证脚本

遍历 data/leave_one/ 下全部物种（104 个），对每个物种执行：
  - 特征数 1~30 × 四种分类模型（LR, CategoricalNB, SVM, RandomForest）
  - 多进程并行（最多 N_CPU 个进程）
  - 按模型分别保存 DataFrame：行=物种，列=feature_num(1~30) + label

用法：
    cd CEP_project
    python scripts/method_compare.py --max-feature 30 --n-cpu 64
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# 项目路径设置
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import LEAVE_ONE_DIR, RESULTS_DIR, N_CPU

# ---------------------------------------------------------------------------
# 模型名称
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    'LogisticRegression',
    'NaiveBayes(Categorical)',
    'SVM',
    'RandomForest',
]


def process_one_species(args):
    """
    对单个物种运行 feature_num in range(1, max_feature+1) × 四种模型。

    Parameters
    ----------
    args : tuple
        (species_id, base_dir, max_feature)

    Returns
    -------
    species_id : str
    train_row  : dict  {model_name: {feature_num: acc, ..., 'label': int}}
    test_row   : dict  {model_name: {feature_num: pred, ..., 'label': int}}
    """
    species_id, base_dir, max_feature = args

    dir_path   = os.path.join(base_dir, species_id)
    feature_df = pd.read_csv(os.path.join(dir_path, 'df_feature.csv'), index_col=0)
    meta_df    = pd.read_csv(os.path.join(dir_path, 'df_meta.csv'),    index_col=0)
    summary_df = pd.read_csv(os.path.join(dir_path, 'df_summary.csv'), index_col=0)

    assert (meta_df.index == feature_df.index).all()
    assert (summary_df.index == feature_df.columns).all()

    true_label = int(meta_df.loc[species_id, 'label'])

    train_species = [s for s in meta_df.index.tolist() if s != species_id]
    train_label   = meta_df.loc[train_species, 'label'].values

    train_row = {m: {'label': true_label} for m in MODEL_NAMES}
    test_row  = {m: {'label': true_label} for m in MODEL_NAMES}

    for feature_num in range(1, max_feature + 1):
        train_feature = feature_df.loc[train_species, :].iloc[:, :feature_num]
        all_feature   = feature_df.iloc[:, :feature_num]

        # Encoding：在全量数据上 fit
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(feature_df.iloc[:, :feature_num])

        X_train = encoder.transform(train_feature)
        X_all   = encoder.transform(all_feature)

        # CategoricalNB 专用：按列 clip 到训练集最大值，防止越界
        X_train_nb     = X_train.astype(int)
        nb_max_per_col = X_train_nb.max(axis=0)
        X_all_nb       = np.clip(X_all, 0, nb_max_per_col).astype(int)

        models = {
            'LogisticRegression':      LogisticRegression(max_iter=1000, random_state=42),
            'NaiveBayes(Categorical)': CategoricalNB(),
            'SVM':                     SVC(kernel='rbf', random_state=42),
            'RandomForest':            RandomForestClassifier(n_estimators=100, random_state=42),
        }

        for model_name, model in models.items():
            Xtr  = X_train_nb if model_name == 'NaiveBayes(Categorical)' else X_train
            Xall = X_all_nb   if model_name == 'NaiveBayes(Categorical)' else X_all

            model.fit(Xtr, train_label)

            train_pred = model.predict(Xtr)
            train_acc  = accuracy_score(train_label, train_pred)

            all_pred    = model.predict(Xall)
            target_pred = int(all_pred[meta_df.index == species_id][0])

            train_row[model_name][feature_num] = train_acc
            test_row[model_name][feature_num]  = target_pred

    return species_id, train_row, test_row


def main():
    parser = argparse.ArgumentParser(
        description='CEP 多模型留一验证（LR, NB, SVM, RF）'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=LEAVE_ONE_DIR,
        help=f'留一验证数据根目录（默认：{LEAVE_ONE_DIR}）'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=os.path.join(RESULTS_DIR, 'method_compare'),
        help=f'结果保存目录（默认：results/method_compare/）'
    )
    parser.add_argument('--max-feature', type=int, default=30,
                        help='最大特征数（默认：30）')
    parser.add_argument('--n-cpu',       type=int, default=N_CPU,
                        help=f'并行进程数（默认：{N_CPU}）')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    species_list = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])
    print(f"共 {len(species_list)} 个物种，使用最多 {args.n_cpu} 个 CPU")

    task_args = [
        (sp, args.base_dir, args.max_feature)
        for sp in species_list
    ]

    with Pool(processes=min(args.n_cpu, len(species_list))) as pool:
        results = pool.map(process_one_species, task_args)

    train_data = {m: {} for m in MODEL_NAMES}
    test_data  = {m: {} for m in MODEL_NAMES}

    for species_id, train_row, test_row in results:
        for m in MODEL_NAMES:
            train_data[m][species_id] = train_row[m]
            test_data[m][species_id]  = test_row[m]

    feat_cols = list(range(1, args.max_feature + 1))
    cols      = feat_cols + ['label']

    for m in MODEL_NAMES:
        train_df = pd.DataFrame(train_data[m]).T[cols]
        train_df.index.name = 'species'
        train_df.to_csv(os.path.join(args.save_dir, f'train_acc_{m}.csv'))

        test_df = pd.DataFrame(test_data[m]).T[cols]
        test_df.index.name = 'species'
        test_df.to_csv(os.path.join(args.save_dir, f'test_pred_{m}.csv'))

        print(f"[{m}] 保存完成")

    print(f"\n所有结果已保存至 {args.save_dir}/")
    print("文件列表:")
    for f in sorted(os.listdir(args.save_dir)):
        print(f"  {f}")


if __name__ == '__main__':
    main()
