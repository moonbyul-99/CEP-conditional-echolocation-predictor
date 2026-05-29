#!/bin/bash
#SBATCH --job-name=msa_700genes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00

# =========================================================================
# MSA 多序列比对脚本（MAFFT）
# 将 data/fasta_717 中的 FASTA 文件批量比对，输出 .aln 文件到 data/msa_output_717/
# 用法（直接运行前请修改下方 INPUT_DIR / OUTPUT_DIR 或使用命令行参数）：
#   bash scripts/msa_alignment.sh [INPUT_DIR] [OUTPUT_DIR]
# =========================================================================

# 默认路径（相对于 CEP_project 根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 可通过命令行参数覆盖
INPUT_DIR="${1:-$PROJECT_ROOT/data/fasta_717}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/data/msa_output_717}"
THREADS=4
MAX_JOBS=32  # 同时运行的最大任务数

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 获取所有fasta文件
fasta_files=($(find $INPUT_DIR -name "*.fasta"))

# 并行处理函数
process_file() {
    fasta_file=$1
    gene_name=$(basename "$fasta_file" .fasta)
    output_file="$OUTPUT_DIR/${gene_name}.aln"
    
    echo "[$(date)] Starting alignment for $gene_name"
    
    # 检查是否已存在结果文件且不为空
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo "[$(date)] Skipping $gene_name - already completed"
        return 0
    fi
    
    # 运行MAFFT
    mafft --thread $THREADS --auto --quiet "$fasta_file" > "$output_file"
    
    # 检查输出文件
    if [ ! -s "$output_file" ]; then
        echo "[$(date)] ERROR: Empty output file for $gene_name"
        rm -f "$output_file"
        return 1
    fi
    
    echo "[$(date)] Completed alignment for $gene_name"
    return 0
}

# 主循环 - 限制同时运行的任务数
job_count=0
for fasta_file in "${fasta_files[@]}"; do
    if [ $job_count -ge $MAX_JOBS ]; then
        # 等待一个任务完成
        wait -n
        job_count=$((job_count-1))
    fi
    
    process_file "$fasta_file" &
    job_count=$((job_count+1))
done

# 等待所有剩余任务完成
wait

echo "[$(date)] All alignments completed!"