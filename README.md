# CEP — Convergent Evolution Prediction

> 基于系统发育序列的趋同性状预测方法（回声定位案例）  
> A phylogeny-aware framework for predicting convergent traits from sequence data.

---

## 项目结构

```
CEP_project/
├── README.md                    # 本文件
├── record.md                    # 重构工作记录
├── cep_draft_0523.pdf           # 论文草稿
├── src/                         # 核心源码
│   ├── config.py                # 全局路径配置（修改此文件适配本机环境）
│   ├── leave_one_eval.py        # CEP 留一验证核心（RF + 计数法）
│   └── esl.py                   # ESL / ESL-PSC 分类器实现
├── scripts/                     # 可执行脚本
│   ├── msa_alignment.sh         # MAFFT 多序列比对（SLURM）
│   ├── preprocess.py            # FASTA → CSV 批量转换
│   ├── leave_one_run.py         # 留一验证批量运行入口
│   └── method_compare.py        # 四模型对比（LR, NB, SVM, RF）
├── data/                        # 数据目录（详见下方）
│   ├── metadata/                # 物种元数据（已复制）
│   ├── fasta_717/               # 717 个基因的原始 fasta（192 个物种）
│   ├── msa_output_717/          # MAFFT 比对结果（.aln 文件）
│   ├── msa_df_717/              # MSA 转换后的 CSV（每基因一个文件）
│   ├── feature_data/            # → 符号链接至 data_717/df_summary/, feature_df/
│   ├── leave_one/               # → 符号链接至 phd_thesis/103_leave/
│   └── leave_one_summary/       # → 符号链接至 data_717/leave_one_summary/
├── notebook/                    # 分析 Notebook（最终版本）
│   ├── 00_基础信息补全.ipynb     # 互信息分布分析
│   ├── 01_谱系差异比对_v2.ipynb  # Prestin (SLC26A5) 序列相似性
│   ├── 02_树状图可视化.ipynb     # 系统发育树可视化
│   ├── 03_序列信息.ipynb         # MSA 位点坐标映射（去 gap → 1-based 坐标，供 AlphaFold 分析）
│   ├── 04_方法对比_v1.ipynb      # 对比方法建模与评估
│   ├── 04_方法对比_v2.ipynb      # CEP 留一验证评估
│   ├── 04_方法对比_v3.ipynb      # CEP 消融实验
│   ├── 05_ppi.ipynb             # Top 基因重要性与 GSEA 富集
│   ├── 07_esl_compare.ipynb     # ESL / ESL-PSC 对比
│   └── 08_eval_plot_v1.ipynb    # 错误案例分析（假阳性/假阴性）
└── results/                     # 输出结果（运行后生成）
```

---

## 环境依赖

### Python 包

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
            tqdm group-lasso Bio
```

### 外部工具

| 工具 | 用途 |
|------|------|
| MAFFT | 多序列比对（MSA），`scripts/msa_alignment.sh` 调用 |
| SLURM（可选） | 集群任务调度，非必需 |

---

## 复现步骤

### Step 0：配置路径

编辑 `src/config.py`，确认各数据目录路径正确。  
默认情况下，`data/` 下的符号链接已指向正确位置，无需修改。

```python
# src/config.py 中的关键变量
CEP_ROOT    # 项目根目录（自动检测）
DATA_DIR    # data/ 目录
METADATA_1_CSV  # 完整元数据（含物种中文名、目分类）
LEAVE_ONE_SUMMARY_DIR  # 留一验证特征排序预计算结果
```

### Step 1：数据预处理

**输入**：OrthoMam v10 下载的 190 物种 CDS FASTA + 鼩鼹 / 软毛树鼠自有数据  
**输出**：`data/msa_df_717/` 下的 CSV 文件（每基因一个文件，行=物种，列=去 gap 后的位点）

在论文分析中总共产生两个版本的MSA结果，第一版是全部的OrthoMam v10 中 190个物种，15k个基因的MSA。
之后我们在基因层面上仅保留存在某个突变位点超过0.35的互信息基因，总计717个（716， 我们删除了某个没有
已知gene symbol的基因）。在这之后我们在fasta中加入了鼩鼹、猪尾鼠的基因，对192个物种的基因重新做MSA。
我们上传的数据仅包含后续192个物种717个基因的原始fasta供复现。第一版的计算结果我们仅保留一个特征互信息
矩阵用于后续MI分布可视化。

fasta_717 中存放717个基因的fasta。 msa_output_717 是msa结果。msa_df_717存放csv格式的结果。

第一版的统计信息在：
df = pd.read_csv('/home/rsun@ZHANGroup.local/hqy_new/not_used_0112/filter_all/summary/res_all.csv', index_col = 0)

```bash
# 1. MAFFT 多序列比对（SLURM 调度，或直接本地运行）
bash scripts/msa_alignment.sh data/fasta_717 data/msa_output_717

# 2. MSA .aln → CSV（preprocess.py 执行以下操作：
#    - 解析 .aln 文件为 species × position DataFrame
#    - 非标准氨基酸替换为 '-'
#    - 按 metadata 对齐物种列表，缺失物种用 '-' 填充
#    - 删除 Homo_sapiens 中 gap 的位点
#    - 列名格式化为 {gene_id} {position}）
python scripts/preprocess.py \
    --fasta-dir data/msa_output_717 \
    --metadata data/metadata/metadata_1.csv \
    --output-dir data/msa_df_717
```

MI 基因级过滤（MI ≥ 0.35，保留 716 个基因）基本逻辑如下：
```python
# idx = df['mi'] >= 0.35
# df_sub = df.loc[idx,:]

# idx = (df_sub.iloc[:,:5] > 0).sum(1) >=3 
# df_sub = df_sub.loc[idx,:]
# df_sub

# tmp = {}
# for ele in df_sub.index:
#     x = ele.split('_')[0]
#     if x in tmp: 
#         continue 
#     else:
#         tmp[x] = 1
# print(len(tmp)) ## 716 个gene list
```

### Step 2：特征分析（Result 1）

| 分析内容 | Notebook | 论文图表 |
|----------|----------|----------|
| 互信息分布 | `notebook/00_基础信息补全.ipynb` | Fig 1b |
| 系统发育树 | `notebook/02_树状图可视化.ipynb` | Fig 1a |
| Prestin 序列相似性 | `notebook/01_谱系差异比对_v2.ipynb` | Fig 1c |
| 10 个位点的跨谱系分布漂移（热力图） | `final_plot/09_count_bar.ipynb` | Fig 1d |
| 趋同突变累积分布（柱状图） | `final_plot/09_count_bar.ipynb` | Fig 1e |


### Step 3：CEP 留一验证

#### 3.1 生成 103_leave 预计算数据

103_leave 目录包含每个物种的 leave-one-out 特征排序预计算结果。

**输入**：
- `data/msa_df_717/*.csv` — Step 1 生成的特征矩阵 CSV 文件
- `data/metadata/metadata_1.csv` — 元数据

**生成脚本**：`scripts/generate_leave_one.py`

```bash
python scripts/generate_leave_one.py \
    --csv-dir data/msa_df_717 \
    --metadata data/metadata/metadata_1.csv \
    --output-dir data/leave_one \
    --top-k 20000 \
    --save-summary \
    --n-cpu 64
```

**生成逻辑**（等价于 `paper_result/01_figure_103.ipynb` + `notebook/07_esl_compare.ipynb` Cell 12）：

1. **构建完整特征矩阵**：读取所有 CSV 文件，拼接为 (104 物种, 728K 位点) 的 DataFrame
2. **计算全局特征排序**：对每个位点计算 NMI、eco_mutation、coverage、score = cover_score × NMI
3. **对每个物种执行 leave-one-out**：
   - 取全局 top 20000 特征（按 score 预筛选）
   - 移除当前物种，用剩余 103 物种重新计算 NMI + coverage + score
   - 取 top 10000 保存

**输出**：`data/leave_one/{species_id}/` 下各三个文件：
- `df_feature.csv` — 特征矩阵（103 物种 × 10000 特征）
- `df_meta.csv` — 元数据（103 行，含 label、order_chinese_new 等）
- `df_summary.csv` — 特征评分（10000 行，含 NMI、eco_cover、score 等）

#### 3.2 运行 CEP 预测

对全部 104 个物种执行 CEP 预测：

```bash
python scripts/leave_one_run.py --top-k 500 --n-cpu 64
```

**CEP 预测策略**（对齐论文 cep_draft_0523.pdf）：
- 翼手目 / 鲸目：使用 **RandomForest**（top 10 特征），返回概率
- 其他目：使用趋同突变计数法（eco_mutation count vs ref_max）

**输出**：`results/logs/cep_leave_one_*.csv`

### Step 4：方法对比与评估（Result 3 & 4）

#### 4.1 脚本一键运行

```bash
# CEP 留一验证（已在 Step 3.2 完成）
python scripts/leave_one_run.py --top-k 500 --n-cpu 64

# 四模型对比（LR / NB / SVM / RF，特征数 1-30）
python scripts/method_compare.py --max-feature 30 --n-cpu 64

# 消融实验（5 种 RF 变体）
python scripts/ablation_study.py --n-cpu 64

# ESL / ESL-PSC 对比（依赖 group-lasso: pip install group-lasso）
python scripts/esl_compare.py --n-cpu 32
```

**输出**：`results/` 目录下
- `logs/cep_leave_one_*.csv` — CEP 预测结果
- `method_compare/` — 四模型的训练准确率和预测结果
- `ablation_study.csv` — 消融实验结果
- `esl_compare.csv` — ESL / ESL-PSC 预测结果

#### 4.2 Notebook 可视化

| 分析内容 | Notebook | 说明 |
|----------|----------|------|
| CEP 评估曲线 + 方法汇总 | `notebook/04_方法对比_v2.ipynb` | 读取脚本输出，绘制准确率曲线 |
| 消融实验可视化 | `notebook/04_方法对比_v3.ipynb` | 对比 5 种 RF 变体 |
| ESL 结果可视化 | `notebook/07_esl_compare.ipynb` | 对比 CEP vs ESL vs ESL-PSC |
| 错误案例分析 | `notebook/08_eval_plot_v1.ipynb` | 分析预测错误的物种 |

### Step 5：基因重要性分析（Result 5）

| 分析内容 | Notebook |
|----------|----------|
| Top 30 基因位点重要性 & GSEA | `notebook/05_ppi.ipynb` |
| PPI 网络（STRING 数据库） | `notebook/05_ppi.ipynb` 生成 `top_gene.txt`，提交 STRING |

---

## 数据说明

### metadata/metadata.csv

基础元数据，191 个物种：
- `split`：数据划分标识
- `label`：回声标签（0=非回声, 1=回声, 2=未知）
- `order`：谱系分组（madaowei / whale / bat / other / mus / unknown）

### metadata/metadata_1.csv

扩展元数据，增加：
- `species_chinese`：物种中文名
- `order_chinese` / `order_chinese_new`：中文目分类

### data/leave_one/ 目录

每个物种一个子目录，包含：
- `df_feature.csv`：特征矩阵（已排序）
- `df_meta.csv`：元数据
- `df_summary.csv`：特征评分汇总

### data/leave_one_summary/ 目录

每个物种一个子目录，包含：
- `feature_nmi.csv`：基于 NMI 排序的特征矩阵
- `df_summary.csv`：特征评分汇总（含 nmi、cover 等列）

---

## 论文草稿

`cep_draft_0523.pdf` 包含完整的 CEP 方法描述与实验结果，供理解整体工作参考。

---

## 引用

如使用本方法，请引用相关论文（待发表后更新）。