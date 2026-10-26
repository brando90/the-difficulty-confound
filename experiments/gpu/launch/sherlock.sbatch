#!/bin/bash
# GPU plans g1-g3 on Sherlock: one array task = one run of the shared registry (common/runs.py).
# From $SCRATCH/tdc_gpu, after sherlock_prep.sbatch has finished (module load python/3.12 first, for runs.py):
#   N=$(python3 the-difficulty-confound/experiments/gpu/common/runs.py count --tier $TIER --plans $PLANS)
#   sbatch --array=0-$((N-1))%8 --export=ALL,TDC_COMMIT=<sha>,TIER=$TIER,PLANS=$PLANS the-difficulty-confound/experiments/gpu/launch/sherlock.sbatch
# Optional --export variables: SEEDS=0 (seed-0 first look), SECONDARY=1 (g3 within-family arms), PY_MODULE, TORCH_INDEX.
# A finished run (DONE) or one claimed by another job exits at once, so resubmitting the same array is safe.
#SBATCH --job-name=tdc_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
# bf16 needs compute capability >= 8.0 (A100, A40, L40S, H100). List GPU types with: sh_node_feat -p gpu | grep GPU_SKU
# If jobs land on older cards and exit with the compute-capability message, add a constraint from that list, e.g.
##SBATCH -C GPU_SKU:A100_SXM4
# For the 1.5B scale tier use 80 GB cards and a longer limit: add  -C GPU_MEM:80GB --time=08:00:00  on the sbatch line.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "submit with sbatch; never compute on a login node" >&2; exit 2; }
ID=${SLURM_ARRAY_TASK_ID:?submit as an array job (--array)}
: "${SCRATCH:?SCRATCH is not set}" "${TIER:?pass TIER=pilot|main|scale}"
PLANS=${PLANS:-g1,g2,g3}
BASE=${BASE:-$SCRATCH/tdc_gpu}
REPO=$BASE/the-difficulty-confound
source "$REPO/experiments/gpu/launch/common_env.sh"
check_commit
[[ -f $DATA/v1/manifest.json ]] || { echo "no data at $DATA; run sherlock_prep.sbatch first" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0) else 1)' ||
  { echo "CUDA unavailable or GPU compute capability < 8.0 (bf16 needs >= 8.0); add a -C GPU constraint (sh_node_feat -p gpu) and resubmit this index" >&2; exit 3; }
EXTRA=()
[[ ${SECONDARY:-0} == 1 ]] && EXTRA+=(--secondary)
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-8}
mkdir -p "$OUT/logs"
cd "$REPO"
START=$(date +%s)
set +e
"$PY" -u "$GPU_DIR/common/train.py" --tier "$TIER" --plans "$PLANS" --seeds "${SEEDS:-all}" ${EXTRA[@]+"${EXTRA[@]}"} --index "$ID" \
  --data-root "$DATA" --out-root "$OUT" >>"$OUT/logs/${TIER}_${PLANS//,/-}_$(printf '%04d' "$ID").log" 2>&1
RC=$?
set -e
printf '%s\t%s\t%s\t%s\tsherlock:%s\t%s\t%s\n' "$(date '+%m-%d-%Y %H:%M:%S')" "$TIER" "$PLANS" "$ID" "$SLURM_JOB_ID" "$RC" "$(($(date +%s) - START))" >>"$OUT/logs/status.tsv"
exit "$RC"
