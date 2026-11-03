#!/bin/bash
# g1 seeds3: one array task = one run of tier seeds3 (common/runs.py list --tier seeds3 --plans g1; 16 runs, index 15 = base).
# Submitted by submit_g1s3.sh from $SCRATCH/tdc_g1s3. H100 80 GB only; ~10 min per run, 25 min cap.
#SBATCH --job-name=g1s3
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
set -uo pipefail
ID=${SLURM_ARRAY_TASK_ID:?submit as an array job}
source "$SCRATCH/tdc_g1s3/the-difficulty-confound/experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/seeds3/env.sh"
[[ -f $VENV/.ready ]] || { echo "venv $VENV is not built (run submit_g1s3.sh on a login node)" >&2; exit 2; }
(cd "$REPO" && sha256sum -c --quiet CODE_SHA256) || { echo "code differs from CODE_SHA256 (the pre-registered commit)" >&2; exit 2; }
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU"
[[ $GPU == *H100* ]] || { echo "not an H100 ($GPU); this check runs on H100 only" >&2; exit 3; }
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-8}
mkdir -p "$OUT/logs"
START=$(date +%s)
"$PY" -u "$GPU_DIR/common/train.py" --tier seeds3 --plans g1 --index "$ID" --data-root "$DATA" --out-root "$OUT" --save-final \
  >>"$OUT/logs/seeds3_$(printf '%02d' "$ID").log" 2>&1
RC=$?
printf '%s\t%s\tsherlock:%s_%s\t%s\t%s\t%s\n' "$(date '+%m-%d-%Y %H:%M:%S')" "$ID" "$SLURM_ARRAY_JOB_ID" "$ID" "$RC" "$(($(date +%s) - START))" "$GPU" >>"$OUT/logs/status.tsv"
bash "$S3/pack_results.sh" >>"$OUT/logs/pack.log" 2>&1   # the report and results tgz, refreshed by every task (seconds)
exit "$RC"
