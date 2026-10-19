#!/bin/bash
# Build the data (pools, evaluation sets, manifest) and pre-download the models, once, on a CPU node.
#   cd $SCRATCH/tdc_gpu && mkdir -p slurm_logs
#   sbatch --export=ALL,TDC_COMMIT=<sha> the-difficulty-confound/experiments/gpu/launch/sherlock_prep.sbatch
#SBATCH --job-name=tdc_gpu_prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "submit with sbatch; never compute on a login node" >&2; exit 2; }
: "${SCRATCH:?SCRATCH is not set}"
BASE=${BASE:-$SCRATCH/tdc_gpu}
REPO=$BASE/the-difficulty-confound
source "$REPO/experiments/gpu/launch/common_env.sh"
check_commit
"$PY" -u "$GPU_DIR/common/prep_data.py" --data-root "$DATA"
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]:
    print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"]))
PYEOF
echo "data and models ready under $BASE"
