#!/bin/bash
# g1 seeds3 re-grade (pre-registered gate failed): one array task = one run re-graded from its saved final weights
# with the pilot's engine (regrade.py --engine pilot). Task 15 = base (hub cache). H100 only. Submit from $SCRATCH/tdc_g1s3.
#SBATCH --job-name=g1s3rg
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
set -uo pipefail
ID=${SLURM_ARRAY_TASK_ID:?submit as an array job}
source "$SCRATCH/tdc_g1s3/the-difficulty-confound/experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/seeds3/env.sh"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU"
[[ $GPU == *H100* ]] || { echo "not an H100 ($GPU)" >&2; exit 3; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-8}
export HF_HUB_OFFLINE=1
RUN=$(ls -d "$OUT"/runs/qwen2.5-0.5b__* | sort | sed -n "$((ID + 1))p")
echo "task $ID -> $RUN"
[[ -d $RUN ]] || { echo "no run dir for index $ID" >&2; exit 2; }
START=$(date +%s)
"$PY" -u "$S3/regrade.py" --run-dir "$RUN" --data-root "$DATA" --engine pilot
RC=$?
echo "$ID $(basename "$RUN") rc=$RC $(($(date +%s) - START))s" >>"$OUT/logs/regrade_status.txt"
exit "$RC"
