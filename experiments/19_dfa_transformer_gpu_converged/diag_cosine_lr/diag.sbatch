#!/bin/bash
# AT-494 diagnostic training on one Sherlock GPU (PREREG.md). Submitted by submit_diag.sh; never run on a login node.
# Budget: the process stops itself at 18 min (control-drop and octave rules); Slurm kills at 24 min (0.4 GPU-h worst case).
#SBATCH --job-name=tdc_diag494
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:24:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "submit with sbatch" >&2; exit 2; }
: "${SCRATCH:?}" "${BASE:?}" "${X19:?}" "${VENV:?}"
OUT=$BASE/diag494_results
source "$X19/launch/env.sh"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
cd "$X19/diag_cosine_lr"
"$PY" -u diag_train.py --out "$OUT" --budget-minutes 18 --device cuda --mask-cache "$BASE/exp19_testmask.npz" \
  --workers "${SLURM_CPUS_PER_TASK:-4}" 2>&1 | tee "$OUT/train.log"
"$PY" diag_summary.py --results "$OUT" | tee "$OUT/summary.log"
cd "$BASE"
tar czf diag494_results.tgz diag494_results/metrics.csv diag494_results/run_state.json diag494_results/manifest.json \
  diag494_results/summary.json diag494_results/gpu.txt diag494_results/train.log diag494_results/summary.log \
  diag494_results/predictions diag494_results/checkpoints
ls -la diag494_results.tgz
echo "DIAG494 DONE"
