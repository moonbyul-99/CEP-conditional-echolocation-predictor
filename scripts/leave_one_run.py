#!/usr/bin/env python3
"""
CEP 留一验证（Leave-One-Species-Out）批量评估入口

对全部 104 个物种执行 CEP 预测：
  - 翼手目/鲸目：使用 RandomForest（top 10 特征）
  - 其他目：使用趋同突变计数法

用法：
    cd CEP_project
    python scripts/leave_one_run.py --top-k 500 --n-cpu 64
"""

import argparse
import os
import sys
import traceback

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import LEAVE_ONE_DIR, N_CPU
from src.leave_one_eval import parallel_processing


def main():
    parser = argparse.ArgumentParser(
        description='CEP 留一验证批量评估（104 物种）'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=LEAVE_ONE_DIR,
        help=f'103_leave 数据根目录（默认：{LEAVE_ONE_DIR}）'
    )
    parser.add_argument('--top-k',   type=int, default=500,
                        help='计数法扫描的最大特征数（默认：500）')
    parser.add_argument('--n-cpu',   type=int, default=N_CPU,
                        help=f'并行进程数（默认：{N_CPU}）')
    args = parser.parse_args()

    try:
        results_df = parallel_processing(
            base_dir=args.base_dir,
            top_k=args.top_k,
            n_cpu=args.n_cpu,
        )
        print(f"\n\033[1;92m✓ 评估完成！详细结果：results/logs/cep_leave_one_*.csv\033[0m")
    except Exception as e:
        print(f"\033[1;91m× 主进程错误: {str(e)}\033[0m")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
