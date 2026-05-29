# CEP 项目重构工作记录

> 生成日期：2026-05-28  
> 目标：将散落在 `hqy_new/` 各处的核心代码整合到 `CEP_project/`，建立可复现的分步流程。

---

## 1. 文件来源映射表

### 核心源码（`src/`）

| 新路径 | 原路径 | 修改内容 |
|--------|--------|----------|
| `src/config.py` | 新建 | 统一管理所有数据路径与计算资源参数 |
| `src/__init__.py` | 新建 | 包初始化，版本号 |
| `src/leave_one_eval.py` | `hqy_new/leave_one_eval.py` | **完全重写**，对齐 `notebook/04_方法对比_v2.ipynb` 中的 `process_species()` 逻辑（详见下方第 7 节） |
| `src/esl.py` | `CEP_project/notebook/final_plot/esl.py` | 直接复制，无修改（原始实现无硬编码路径） |
| ~~`src/model.py`~~ | ~~`hqy_new/model.py`~~ | **已删除**：早期 AMI 特征排序，论文已改用 cover_score × NMI，全项目无引用 |
| ~~`src/bayes_model.py`~~ | ~~`hqy_new/bayes_model.py`~~ | **已删除**：旧版朴素贝叶斯预测，论文已改用 RandomForest，全项目无引用 |
| ~~`src/logger.py`~~ | ~~`hqy_new/logger.py`~~ | **已删除**：重写 leave_one_eval.py 后未再使用，全项目无引用 |

### 脚本（`scripts/`）

| 新路径 | 原路径 | 修改内容 |
|--------|--------|----------|
| `scripts/preprocess.py` | `hqy_new/src/seq2df.py` | 重写为带 argparse 的命令行工具；路径默认值从 `config.py` 读取 |
| `scripts/leave_one_run.py` | `hqy_new/leave_one_run.py` | 改为 argparse 入口；import 路径改为 `from src.leave_one_eval import` |
| `scripts/method_compare.py` | `hqy_new/phd_thesis/04_method_compare.py` | 改为 argparse 入口；`BASE_DIR` 改为 `config.LEAVE_ONE_DIR`；`SAVE_DIR` 改为 `config.RESULTS_DIR/method_compare` |
| `scripts/msa_alignment.sh` | `CEP_project/scripts/msa_alignment.sh`（已有） | 硬编码路径改为相对项目根目录，支持命令行参数覆盖 |

---

## 2. 数据目录（符号链接）

| 链接路径 | 指向目标 | 说明 |
|----------|----------|------|
| `data/fasta_717/` | 实际数据目录 | 717 个基因的原始 fasta（192 个物种，含鼩鼹、猪尾鼠） |
| `data/msa_output_717/` | 实际数据目录 | MAFFT 比对结果（.aln 文件） |
| `data/msa_df_717/` | 实际数据目录 | MSA 转换后的 CSV（每基因一个文件，由 `preprocess.py` 生成） |
| `data/feature_data/df_summary/` | `hqy_new/data_717/df_summary/` | 全局特征汇总 |
| `data/feature_data/feature_df/` | `hqy_new/data_717/feature_df/` | 全局特征矩阵 |
| `data/leave_one/` | `hqy_new/phd_thesis/103_leave/` | 104 个物种的留一验证数据（方法对比用） |
| `data/leave_one_summary/` | `hqy_new/data_717/leave_one_summary/` | 104 个物种的特征排序预计算结果（CEP 评估用） |
| `data/metadata/metadata.csv` | 复制自 `hqy_new/metadata/metadata.csv` | 基础元数据 |
| `data/metadata/metadata_1.csv` | 复制自 `hqy_new/metadata/metadata_1.csv` | 扩展元数据（含中文名） |
| `data/metadata/idx2gene.txt` | 复制自 `hqy_new/metadata/idx2gene.txt` | 基因索引映射 |
| `data/metadata/args_train.json` | 复制自 `hqy_new/metadata/args_train.json` | 留一验证参考物种列表 |
| `data/metadata/mapdic.json` | 复制自 `hqy_new/metadata/mapdic.json` | 映射字典 |

---

## 3. 删除的文件及原因

### Notebook 冗余版本

| 删除文件 | 保留版本 | 原因 |
|----------|----------|------|
| `notebook/01_谱系差异比对.ipynb` | `01_谱系差异比对_v2.ipynb` | 旧版本，v2 为最终版 |
| `notebook/01_谱系差异比对_v1.ipynb` | `01_谱系差异比对_v2.ipynb` | 旧版本，v2 为最终版 |
| `notebook/04_方法对比.ipynb` | `04_方法对比_v1/v2/v3.ipynb` | 旧版本，v1/v2/v3 分工更明确 |
| `notebook/08_eval_plot.ipynb` | `08_eval_plot_v1.ipynb` | 旧版本，v1 为最终版 |
| `notebook/05_predict_all.ipynb` | — | 功能已被 `04_方法对比_v2.ipynb` 覆盖 |
| `notebook/06_特征选择.ipynb` | — | 功能已被 `04_方法对比_v3.ipynb` 覆盖 |
| `notebook/09_count_bar.ipynb` | — | 功能已被 `08_eval_plot_v1.ipynb` 覆盖 |
| `notebook/10_esl_cep_compare.ipynb` | `07_esl_compare.ipynb` | ESL 对比已整合到 07 |
| `notebook/create_717gene.ipynb` | `scripts/preprocess.py` | 已有独立的 .py 脚本替代 |

### 空目录

| 删除路径 | 原因 |
|----------|------|
| `data/filter_data/` | 空目录，无数据 |

---

## 4. 保留的 Notebook 清单与功能说明

| Notebook | 对应论文图表 | 功能 |
|----------|--------------|------|
| `00_基础信息补全.ipynb` | Fig 1b | 互信息分布分析 + MI ≥ 0.35 基因级过滤，~9M 突变位点 |
| `01_谱系差异比对_v2.ipynb` | Fig 1c | Prestin (SLC26A5) 序列相似性分析 |
| `02_树状图可视化.ipynb` | Fig 1a | 系统发育树可视化 |
| `03_序列信息.ipynb` | 辅助分析 | MSA 位点坐标映射：将对齐位置映射到去 gap 后的 1-based 坐标（供 AlphaFold 使用） |
| `04_方法对比_v1.ipynb` | Result 3 (Table) | 四种分类模型留一验证建模 |
| `04_方法对比_v2.ipynb` | Fig 3 / Fig 4 | CEP 留一验证评估 + 消融实验 + 方法对比 |
| `04_方法对比_v3.ipynb` | Fig 3 | CEP 消融实验（详细版） |
| `05_ppi.ipynb` | Fig 5 | Top 基因重要性分布（Manhattan 图）+ GSEA 富集 + PPI 网络 |
| `07_esl_compare.ipynb` | Fig 4 (Table) | ESL / ESL-PSC 方法对比 |
| `08_eval_plot_v1.ipynb` | 补充分析 | 错误案例分析（假阳性/假阴性物种） |

---

## 5. 路径修改详细记录

### `src/bayes_model.py`

```
# 原始（硬编码）
tmp_meta = pd.read_csv('/home/rsun@ZHANGroup.local/hqy_new/metadata/metadata_1.csv', index_col=0)

# 修改后（参数化 + config 默认值）
if metadata_1_csv is None:
    metadata_1_csv = _DEFAULT_METADATA_1_CSV  # from src.config
tmp_meta = pd.read_csv(metadata_1_csv, index_col=0)
```

### `src/leave_one_eval.py`

```
# 原始硬编码路径（共 4 处）：
'/home/rsun@ZHANGroup.local/hqy_new/metadata/metadata_1.csv'
f'/home/rsun@ZHANGroup.local/hqy_new/data_717/leave_one_summary/{leave_species}/feature_nmi.csv'
f'/home/rsun@ZHANGroup.local/hqy_new/data_717/leave_one_summary/{leave_species}/df_summary.csv'
"logs"  # 日志目录

# 修改后：
from src.config import METADATA_1_CSV, LEAVE_ONE_SUMMARY_DIR, LOGS_DIR
# 所有路径均从 config.py 读取
```

### `scripts/msa_alignment.sh`

```bash
# 原始（硬编码）
INPUT_DIR="/home/rsun@ZHANGroup.local/hqy_new/new_fasta"
OUTPUT_DIR="/home/rsun@ZHANGroup.local/hqy_new/new_fasta_msa"

# 修改后（相对路径 + 命令行参数覆盖）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="${1:-$PROJECT_ROOT/data/new_fasta}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/data/new_fasta_msa}"
```

---

## 6. 注意事项

1. **符号链接依赖**：`data/raw_data/`、`data/leave_one/` 等目录为符号链接，指向 `hqy_new/` 下的实际数据。  
   - 若将 CEP_project 复制到其他机器，需同步迁移数据并重建链接。
   - 重建链接命令示例：`ln -sfn /new/path/to/data CEP_project/data/raw_data`

2. **metadata 文件为副本**：`data/metadata/` 下的文件是复制品（非链接），与 `hqy_new/metadata/` 独立。  
   - 若原始 metadata 有更新，需手动同步到 `CEP_project/data/metadata/`。

3. **config.py 是路径配置入口**：所有 Python 脚本的数据路径均通过 `src/config.py` 管理。  
   - 在新环境中运行，优先检查 `config.py` 中的路径是否有效。

4. **Notebook 路径未更新**：保留的 notebook 中仍有部分硬编码的绝对路径（指向 `hqy_new/` 下的原始位置）。  
   - 符号链接的存在使得大部分 notebook 通过 `data/` 目录可正常访问数据。
   - 部分 notebook 引用外部数据（如 `not_used_0112/filter_all/summary/res_all.csv`），需在 notebook 中手动确认路径。

5. **`data/leave_one/` 与 `data/leave_one_summary/` 是两套不同数据**：
   - `leave_one/`（来自 `phd_thesis/103_leave/`）：用于 `src/leave_one_eval.py`（CEP 评估）和 `scripts/method_compare.py`（四模型对比），包含 `df_feature.csv`、`df_meta.csv`、`df_summary.csv`
   - `leave_one_summary/`（来自 `data_717/leave_one_summary/`）：早期版本的特征排序预计算结果，包含 `feature_nmi.csv`、`df_summary.csv`

6. **未整合的部分**：
   - `not_used_0112/filter_all/summary/res_all.csv`（~9M 突变位点的 MI 分析）：文件较大，`00_基础信息补全.ipynb` 中直接引用原路径
   - `data_717/717feature_align/`（716 个基因的 feature alignment）：预处理中间产物，未建链接
   - `hqy_new/CEP_project/notebook/final_plot/`（论文最终图表 SVG/PDF）：保留在原位，notebook 运行后会重新生成
   - `paper_result/summary_df.csv`（728K 个位点的全局特征排序）：由 `paper_result/01_figure_103.ipynb` 生成，用于 103_leave 预计算的初始特征筛选

---

## 8. 数据生成流程说明（2026-05-28 补充）

### 8.1 全局特征排序（summary_df.csv）

**代码位置**：`scripts/generate_leave_one.py`（等价于 `paper_result/01_figure_103.ipynb`）

**输入**：
- `data/msa_df_717/*.csv` — 716 个基因的特征矩阵 CSV 文件
- `data/metadata/metadata_1.csv`

**输出**：`results/summary_df.csv`（728K 行，包含 NMI、Eco_Mode、coverage、cover_score、score 等列）

**用法**：`python scripts/generate_leave_one.py --save-summary`

### 8.2 103_leave 预计算数据

**代码位置**：`scripts/generate_leave_one.py`（等价于 `notebook/07_esl_compare.ipynb` Cell 12）

**输入**：
- `data/msa_df_717/*.csv` — 716 个基因的特征矩阵 CSV 文件
- `data/metadata/metadata_1.csv`

**输出**：`data/leave_one/{species_id}/` 下的 `df_feature.csv`、`df_meta.csv`、`df_summary.csv`

**生成逻辑**：
1. 读取所有 CSV，拼接为完整特征矩阵 (104 物种 × 728K 位点)
2. 计算全局特征排序 (NMI + coverage + score)
3. 取 top 20000 特征，对每个物种执行 leave-one-out，保存 top 10000

---

## 9. Step 4 新增脚本（2026-05-29 补充）

为了让 Step 4 的方法对比与评估能够一键运行，新增以下脚本：

### 9.1 scripts/esl_compare.py

**等价于**：`notebook/07_esl_compare.ipynb` Cell 3 + Cell 4 + Cell 6

**功能**：对全部 104 个物种执行 ESL 和 ESL-PSC 留一预测
- 从 `data/msa_df_717/*.csv` 重建完整特征矩阵
- 每个基因取 NMI 最高的 100 个位点
- 运行 ESLClassifier 和 ESLPSCClassifier
- 输出 `results/esl_compare.csv`

**依赖**：`pip install group-lasso`

### 9.2 scripts/ablation_study.py

**等价于**：`notebook/04_方法对比_v3.ipynb`

**功能**：运行 5 种 RF 变体对比实验

| 变体 | 训练集 | 特征排序 | 对应 notebook 章节 |
|------|--------|----------|-------------------|
| `local_no_cold` | 同目物种 | score×NMI | “local RF 无冷启动” |
| `global_rf` | 全部物种 | score×NMI | “全局RF” |
| `coldstart_score` | 冷启动 | score×NMI | “LOCAL RF + 冷启动” |
| `coldstart_random` | 冷启动 | 随机 | “LOCAL RF + 冷启动 + 随机特征” |
| `coldstart_nmi` | 冷启动 | NMI | “LOCAL RF + 冷启动 + NMI” |

- 输出 `results/ablation_study.csv`

---

## 7. CEP 评估逻辑修复记录（2026-05-28 补充）

> 针对用户指出的三个问题，对 `src/leave_one_eval.py` 和 `src/model.py` 进行了完全重写。

### 问题 1：ZWS / shrew_mole 被错误过滤

**旧版错误代码**（`hqy_new/leave_one_eval.py` 中的 `load_data()`）：
```python
idx = meta_df.index.isin(['ZWS', 'shrew_mole', leave_species])
meta_df = meta_df.loc[~idx, :]  # 将 ZWS 和 shrew_mole 从数据中移除
```

**问题**：ZWS（猪尾鼠）和 shrew_mole（鼩鼹）都是回声物种（label=1），需要参与 104 物种的留一评估。旧版将其强制过滤导致这两个物种无法被评估。

**修复**：新版 `leave_one_eval.py` 数据来源改为 `data/leave_one/{species_id}/` 目录（每物种一个目录），其中 `df_meta.csv` 包含全部 104 物种。仅过滤 `label==2`（未知标签）的物种。

### 问题 2：贝叶斯改为 RandomForest

**旧版错误代码**：
```python
if leave_order in (['翼手目', '鲸目']):
    plot_df, score, ref_weight = bayes_predict(...)  # 调用 CategoricalNB
```

**问题**：论文使用 RandomForest 作为 base model，旧版对翼手目/鲸目调用朴素贝叶斯与论文不符。

**修复**：新版 `predict_species()` 严格对齐 notebook `process_species()` 的逻辑：
- 翼手目 / 鲸目：`RandomForestClassifier(n_estimators=100)`，top 10 特征，`OrdinalEncoder` 编码
- 其他目：趋同突变计数法（eco_mutation count vs ref_max，含 7% 阈值过滤）

### 问题 3：model.py 文档注释错误

**旧版错误注释**：
```python
### 预测算法：
#   1. 判断预测物种所在的目是否在训练数据中的数目。
#      如果数目 >= 5 并且包含回声+非回声，执行朴素贝叶斯算法
#   2. 统计预测物种的回声趋同突变数目是否超过阈值
#   3. 统计近邻列表中的物种的回声趋同数目的最大值
```

**问题**：描述与实际 CEP 算法完全不符。

**修复**：重写 `model.py` 文档，明确标注为“早期版本的特征排序模块”，当前 CEP 的排序和预测逻辑分别在 `leave_one_eval.py` 的 `rescore_features()` 和 `predict_species()` 中实现。

### 新版 leave_one_eval.py 核心函数对照表

| 函数 | 功能 | 对齐 notebook |
|------|------|---------------|
| `load_species_data()` | 加载 103_leave/{species_id}/ 数据 | `process_species()` 前 3 行 |
| `rescore_features()` | cover_score × NMI 重排 + Eco_Mode 过滤 | `score_map_4/5` + `Eco_Mode != '-'` |
| `build_ref_list()` | 根据目构建参考物种列表 | 全部 if/elif 分支 |
| `predict_species()` | RF（翼手目/鲸目）或计数法（其他） | RandomForestClassifier / count |
| `parallel_processing()` | 多进程批量评估 104 物种 | notebook 中 `Pool` 调用 |

### 验证结果（5 个代表性物种）

| 物种 | 目 | 真实标签 | 预测 | 概率 | 方法 |
|------|------|----------|------|------|------|
| ZWS（猪尾鼠） | 啮齿目 | 回声(1) | 1 | 0.71 | 计数法 |
| 蓝鲸 | 鲸目 | 非回声(0) | 0 | 0.07 | RF(10特征) |
| 马铁菊头蝠 | 翼手目 | 回声(1) | 1 | 0.82 | RF(10特征) |
| 鼩鼹 | 真盲缺目 | 回声(1) | 1 | 0.57 | 计数法 |
| 黄牛 | 偶蹄目 | 非回声(0) | 0 | 0.07 | 计数法 |
