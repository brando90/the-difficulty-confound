#!/bin/bash
# Experiment 19 training on one Sherlock GPU. Submitted by submit_at445.sh; never run on a login node.
# The GPU-time cap is set here, at job start, from the GPU hours AT-441's pilot actually used (PROTOCOL Section 3.3).
#SBATCH --job-name=tdc_exp19
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "submit with sbatch" >&2; exit 2; }
: "${SCRATCH:?}" "${BASE:?}" "${X19:?}"
source "$X19/launch/env.sh"
CAPS=$("$PY" "$X19/launch/budget.py" --at441-jobs "$BASE/jobs_at441.txt" --config "$X19/expt_v1/config.json")
echo "$CAPS"
CAP=$(echo "$CAPS" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["cap_minutes"])')
mkdir -p "$OUT"
echo "$CAPS" >"$OUT/budget_at_start.json"
if "$PY" -c "import sys; sys.exit(0 if float('$CAP') >= 30 else 1)"; then :; else
  echo "GPU budget left is under 30 minutes; not training" | tee "$OUT/NOT_RUN.txt"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
cd "$X19/expt_v1"
# The cap counts from here; the budget includes building the train/test lookup table (about a minute on 8 CPUs).
"$PY" -u gpu_train.py --out "$OUT" --budget-minutes "$CAP" --device cuda --mask-cache "$BASE/exp19_testmask.npz" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" 2>&1 | tee "$OUT/train.log"
