# CEP — Convergent Evolution Prediction

[English](README.md) | [中文](README_CN.md)

> **Decoding Evolutionary Repeatability for Echolocation Prediction**
>
> A phylogeny-aware framework for predicting convergent traits from sequence data.

![CEP Framework](cep.png)

---

The CEP framework takes multiple sequence alignments (MSAs) and a phenotypically labeled phylogenetic tree as input. It first performs proteome-wide feature selection to isolate high-SNR convergent sites, then builds lineage-specific prediction models to control for phylogenetic confounding. Outputs include a robust trait prediction model and prioritized molecular signals for downstream functional analysis.

> **Note**: Due to time and resource constraints, this repository contains Chinese comments and Chinese metadata throughout the codebase and notebooks. We apologize for any inconvenience this may cause.

---

## Project Structure

```
CEP_project/
├── README.md                         # This file (English)
├── README_CN.md                      # Chinese version
├── cep.png                           # Framework diagram
├── cep_draft_0523.pdf                # Manuscript draft
│
├── src/                              # Core source code
│   ├── config.py                     # Global paths & parameters
│   ├── leave_one_eval.py             # CEP leave-one-out core algorithm
│   ├── esl.py                        # ESL & ESL-PSC classifier implementation
│   └── __init__.py                   # Package init
│
├── scripts/                          # Executable scripts
│   ├── msa_alignment.sh              # MAFFT batch MSA alignment
│   ├── preprocess.py                 # FASTA → CSV batch conversion
│   ├── generate_leave_one.py         # Generate leave-one-out precomputed data
│   ├── leave_one_run.py              # CEP leave-one-out batch run
│   ├── method_compare.py             # 4-model baseline (LR, NB, SVM, RF)
│   ├── ablation_study.py             # CEP ablation study (5 variants)
│   ├── esl_eval.py                   # ESL leave-one-out evaluation
│   ├── esl_psc_eval.py               # ESL-PSC leave-one-out evaluation
│   └── eval_summary.py               # One-click evaluation summary & plots
│
├── notebook/                         # Analysis notebooks (figure reproduction)
│   ├── Fig_1a.ipynb                  # Fig 1a — Phylogenetic tree
│   ├── Fig_1b.ipynb                  # Fig 1b — MI distribution
│   ├── Fig_1c.ipynb                  # Fig 1c — Prestin sequence similarity
│   ├── Fig_1_de.ipynb                # Fig 1d–e — Convergent mutation accumulation
│   ├── Fig_3_and_Fig4.ipynb          # Fig 3–4 — Method comparison & evaluation
│   ├── Fig_5.ipynb                   # Fig 5 — Top gene enrichment & PPI
│   ├── ori_cep.ipynb                 # Original CEP (reference)
│   ├── ori_esl.ipynb                 # Original ESL/ESL-PSC (reference)
│   ├── ori_ml.ipynb                  # Original ML baseline (reference)
│   └── ori_ablation.ipynb            # Original ablation study (reference)
│
├── data/                             # Data directory
│   ├── metadata/                     # Species metadata
│   │   ├── metadata.csv              #   Basic metadata (191 species)
│   │   ├── metadata_1.csv            #   Extended metadata (Chinese names, orders)
│   │   ├── idx2gene.txt              #   Feature index → gene name
│   │   ├── args_train.json           #   Training parameters
│   │   └── mapdic.json               #   Species name mapping
│   ├── fasta_717/                    # 717 gene FASTA files (192 species)
│   ├── msa_output_717/               # MAFFT alignment outputs (.aln)
│   ├── msa_df_717/                   # CSV-converted MSAs (one per gene)
│   ├── feature_data/                 # Feature matrix cache (Parquet)
│   └── leave_one/                    # Precomputed leave-one-out data (104 dirs)
│
├── results/                          # Output results
│   ├── logs/                         #   CEP prediction logs
│   ├── method_compare/               #   4-model predictions & train accuracies
│   ├── ablation_study.csv            #   Ablation study results
│   ├── esl_eval.csv                  #   ESL predictions
│   ├── esl_psc_eval.csv              #   ESL-PSC predictions
│   ├── compare_eval_plot.svg         #   Method comparison heatmap
│   ├── ablation_eval_plot.svg        #   Ablation study heatmap
│   ├── feature_count_*.svg           #   Per-model feature-count analysis (×4)
│   ├── all_methods_pred.csv          #   All-method per-species predictions
│   ├── all_methods_metrics.csv       #   All-method metrics summary
│   └── all_methods_errors.csv        #   All-method error species
│
└── .gitignore
```

---

## Environment

### Python Packages

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
            tqdm group-lasso Bio
```

### External Tools

| Tool | Purpose |
|------|------|
| MAFFT | Multiple sequence alignment |

---

## Reproduction Steps

### Step 0: Configure Paths

Edit `src/config.py` to verify data directory paths.

```python
# src/config.py — Key variables
CEP_ROOT     # Project root (auto-detected)
DATA_DIR     # data/ directory
N_CPU = 64   # Default CPU count for multiprocessing
```

---

### Step 1: Data Preprocessing

**Input**: OrthoMam v10 (190 species CDS) + two additional species  
**Output**: CSV files under `data/msa_df_717/` (one per gene, rows=species, cols=de-gapped positions)

> Two versions of MSA results were generated during the analysis. Version 1 contains all ~15k genes from 190 OrthoMam v10 species. Version 2 retains only 716 genes with MI ≥ 0.35 at any position, adds two additional species, and realigns 192 species. This repository provides only the version 2 data (717 gene FASTAs) for reproduction.

```bash
# 1. MAFFT alignment
bash scripts/msa_alignment.sh data/fasta_717 data/msa_output_717

# 2. MSA .aln → CSV
#    - Parse .aln to species × position DataFrame
#    - Replace non-standard amino acids with '-'
#    - Drop positions that are gaps in Homo_sapiens
#    - Column name format: {gene_id}_{position}
python scripts/preprocess.py \
    --fasta-dir data/msa_output_717 \
    --metadata data/metadata/metadata_1.csv \
    --output-dir data/msa_df_717
```

---

### Step 2: Feature Analysis — Result 1

Notebooks for reproducing Fig 1:

| Analysis | Notebook | Figure |
|----------|----------|--------|
| Phylogenetic tree | `notebook/Fig_1a.ipynb` | Fig 1a |
| MI distribution | `notebook/Fig_1b.ipynb` | Fig 1b |
| Prestin sequence similarity | `notebook/Fig_1c.ipynb` | Fig 1c |
| Convergent mutation accumulation | `notebook/Fig_1_de.ipynb` | Fig 1d–e |

---

### Step 3: CEP Leave-One-Out Validation

#### 3.1 Generate Leave-One-Out Data

Precompute leave-one-out feature rankings for each species.

```bash
python scripts/generate_leave_one.py \
    --csv-dir data/msa_df_717 \
    --metadata data/metadata/metadata_1.csv \
    --output-dir data/leave_one \
    --top-k 20000 \
    --save-summary \
    --n-cpu 64
```

**Output** (`data/leave_one/{species_id}/`):  
- `df_feature.csv` — Feature matrix (103 species × top-K features)  
- `df_meta.csv` — Metadata (103 rows: label, order_chinese_new)  
- `df_summary.csv` — Feature scores (NMI, eco_cover, score)

#### 3.2 Run CEP Prediction

```bash
python scripts/leave_one_run.py --top-k 500 --n-cpu 64
```

**CEP Prediction Strategy**:
- Chiroptera / Cetacea: **RandomForest** (top 10 features, n_estimators=100)
- Other orders: Convergent mutation counting (eco_mutation count vs ref_max)
- Feature ranking: cover_score × NMI

**Output**: `results/logs/cep_leave_one_*.csv`

---

### Step 4: Method Comparison & Evaluation — Result 3 & 4

#### 4.1 One-Click Script Execution

```bash
# CEP leave-one-out (Step 3.2)
python scripts/leave_one_run.py --top-k 500 --n-cpu 64

# 4-model baseline (LR / NB / SVM / RF, 1–30 features)
python scripts/method_compare.py --max-feature 30 --n-cpu 64

# Ablation study (5 CEP variants)
python scripts/ablation_study.py --n-cpu 64

# ESL evaluation
python scripts/esl_eval.py --n-cpu 32

# ESL-PSC evaluation
python scripts/esl_psc_eval.py --n-cpu 32
```

#### 4.2 Evaluation Summary (One-Click)

```bash
# Collect all results → compute metrics → generate heatmaps
python scripts/eval_summary.py
```

**Output**:

| File | Description |
|------|-------------|
| `compare_eval_plot.svg` | Method comparison heatmap (CEP / ESL / ESL-PSC / RF / LR / NB / SVM) |
| `ablation_eval_plot.svg` | Ablation study heatmap (5 CEP variants) |
| `feature_count_*.svg` | Per-model feature-count analysis (4 files: errors & metrics across 1–30 features) |
| `all_methods_pred.csv` | Per-species predictions for all methods |
| `all_methods_metrics.csv` | Accuracy / Precision / Recall / F1 per method |
| `all_methods_errors.csv` | Error species per method |

#### 4.3 Notebook Visualization

| Analysis | Notebook |
|----------|----------|
| Method comparison & CEP evaluation | `notebook/Fig_3_and_Fig4.ipynb` |
| ESL / ESL-PSC results | `notebook/ori_esl.ipynb` |

---

### Step 5: Gene Importance Analysis — Result 5

| Analysis | Notebook |
|----------|----------|
| Top-30 gene site importance & GSEA enrichment | `notebook/Fig_5.ipynb` |
| PPI network (STRING database) | `notebook/Fig_5.ipynb` |

---

## Data Description

### metadata/metadata.csv

Basic metadata (191 species):
- `split`: Data split label
- `label`: Echolocation label (0 = non-echolocating, 1 = echolocating, 2 = unknown)
- `order`: Lineage grouping

### metadata/metadata_1.csv

Extended metadata with:
- `species_chinese`: Chinese species name
- `order_chinese` / `order_chinese_new`: Chinese order classification

### data/leave_one/

One subdirectory per species, containing:
- `df_feature.csv`: Feature matrix (sorted)
- `df_meta.csv`: Metadata
- `df_summary.csv`: Feature scoring summary

---

## Citation

If you use this method, please cite the corresponding paper (to be updated upon publication).
