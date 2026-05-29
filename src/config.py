"""
CEP Project — 全局路径配置

所有数据路径在此统一管理，其余模块通过 `from src.config import ...` 引用。
如需在本机运行，只需修改此文件中的路径即可，无需改动其他代码。

目录结构约定：
    CEP_project/
    └── data/
        ├── metadata/         # metadata.csv, metadata_1.csv, idx2gene.txt, args_train.json, mapdic.json
        ├── fasta_717/        # 717 个基因的原始 fasta（192 个物种）
        ├── msa_output_717/   # MAFFT 比对结果（.aln 文件）
        ├── msa_df_717/       # MSA 转换后的 CSV（每基因一个文件，行=物种，列=去 gap 后的位点）
        ├── feature_data/     # 特征矩阵与汇总（df_summary/, feature_df/）
        ├── leave_one/        # 留一验证预计算数据（103 个物种目录）
        └── leave_one_summary/# 留一验证特征排序结果（每物种一个目录）
"""

import os

# ---------------------------------------------------------------------------
# 项目根目录：src/ 的父目录，即 CEP_project/
# ---------------------------------------------------------------------------
CEP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 数据目录
# ---------------------------------------------------------------------------
DATA_DIR              = os.path.join(CEP_ROOT, 'data')
METADATA_DIR          = os.path.join(DATA_DIR, 'metadata')
FASTA_717_DIR         = os.path.join(DATA_DIR, 'fasta_717')
MSA_OUTPUT_717_DIR    = os.path.join(DATA_DIR, 'msa_output_717')
MSA_DF_DIR            = os.path.join(DATA_DIR, 'msa_df_717')
FEATURE_DATA_DIR      = os.path.join(DATA_DIR, 'feature_data')
LEAVE_ONE_DIR         = os.path.join(DATA_DIR, 'leave_one')
LEAVE_ONE_SUMMARY_DIR = os.path.join(DATA_DIR, 'leave_one_summary')

# ---------------------------------------------------------------------------
# 常用元数据文件
# ---------------------------------------------------------------------------
METADATA_CSV    = os.path.join(METADATA_DIR, 'metadata.csv')
METADATA_1_CSV  = os.path.join(METADATA_DIR, 'metadata_1.csv')
IDX2GENE_TXT    = os.path.join(METADATA_DIR, 'idx2gene.txt')
ARGS_TRAIN_JSON = os.path.join(METADATA_DIR, 'args_train.json')
MAPDIC_JSON     = os.path.join(METADATA_DIR, 'mapdic.json')

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(CEP_ROOT, 'results')
LOGS_DIR    = os.path.join(CEP_ROOT, 'results', 'logs')

# ---------------------------------------------------------------------------
# 计算资源
# ---------------------------------------------------------------------------
N_CPU = 64  # 多进程默认使用的 CPU 数
