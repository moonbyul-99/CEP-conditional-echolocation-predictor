#!/usr/bin/env python3
"""
数据预处理脚本：将 MSA 对齐结果（FASTA 或 MAFFT .aln 格式）转换为 CSV

处理流程：
1. 解析 .fasta / .aln 文件为 species × mutation 的 DataFrame（支持 MAFFT 多行格式，自动拼接续行）
2. 将 20 种常见氨基酸之外的字符替换为 '-'
3. 根据 Homo_sapiens 位置过滤，删除 human 中 gap 的位点
4. 根据 metadata 过滤不属于元数据的物种，缺失物种用 '-' 填充
5. 添加列名 {gene_id} {position}，保存为 CSV

用法：
    python scripts/preprocess.py \
        --fasta-dir data/msa_output_717 \
        --metadata data/metadata/metadata_1.csv \
        --output-dir data/msa_df_717 \
        --n-cpu 64
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings

# ---------------------------------------------------------------------------
# 项目路径设置
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import METADATA_1_CSV, MSA_DF_DIR, N_CPU

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
AA_LIST    = list("-ACDEFGHIKLMNPQRSTVWY")
GAP_SYMBOL = '-'


def single_process(args):
    """
    单个 FASTA 文件处理函数（供多进程调用）。

    Parameters
    ----------
    args : tuple
        (file_path, meta_df_path, save_dir)
    """
    file_path, meta_df_path, save_dir = args

    meta_df = pd.read_csv(meta_df_path, index_col=0)

    species_list = []
    seq_parts    = []  # 当前物种累积的序列片段
    cur_seq      = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                # 保存上一个物种的序列
                if species_list:
                    seq_parts.append(cur_seq)
                species_list.append(line[1:].strip())
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        # 保存最后一个物种
        if species_list:
            seq_parts.append(cur_seq)

    # 将片段列表拼接为完整序列字符串
    seq_info = [list(''.join(parts)) for parts in seq_parts]

    if 'Homo_sapiens' not in species_list:
        warnings.warn(f"Skipping {file_path}: Homo_sapiens not found.")
        return

    df = pd.DataFrame(seq_info, index=species_list)
    df_std = df.reindex(meta_df.index, fill_value=GAP_SYMBOL)
    df_std[~df_std.isin(AA_LIST)] = GAP_SYMBOL

    x        = df_std.loc['Homo_sapiens', :].values
    gap_idx  = np.where(x == GAP_SYMBOL)[0]
    df       = df_std.drop(gap_idx, axis=1)

    gene_id    = Path(file_path).stem
    df.columns = [f'{gene_id} {i}' for i in range(df.shape[1])]

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'{gene_id}.csv'))


def main():
    parser = argparse.ArgumentParser(
        description='将 FASTA 序列文件批量转换为 CSV（species × positions）'
    )
    parser.add_argument(
        '--fasta-dir',
        type=str,
        required=True,
        help='包含 .fasta 文件的目录路径'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=METADATA_1_CSV,
        help=f'元数据 CSV 文件路径（默认：{METADATA_1_CSV}）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=MSA_DF_DIR,
        help=f'输出目录（默认：{MSA_DF_DIR}）'
    )
    parser.add_argument(
        '--n-cpu',
        type=int,
        default=N_CPU,
        help=f'并行进程数（默认：{N_CPU}）'
    )
    args = parser.parse_args()

    fasta_files = [
        os.path.join(args.fasta_dir, f)
        for f in os.listdir(args.fasta_dir)
        if f.endswith('.fasta') or f.endswith('.aln')
    ]
    if not fasta_files:
        print(f"未在 {args.fasta_dir} 中找到 .fasta 或 .aln 文件。")
        sys.exit(1)

    print(f"共找到 {len(fasta_files)} 个 FASTA 文件，使用 {args.n_cpu} 个 CPU 处理。")

    task_args = [
        (fp, args.metadata, args.output_dir)
        for fp in fasta_files
    ]

    num_workers = min(args.n_cpu, cpu_count())
    with Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(single_process, task_args),
            total=len(task_args),
            desc="Converting FASTA → CSV"
        ))

    print(f"完成！结果保存在：{args.output_dir}")


if __name__ == '__main__':
    main()
