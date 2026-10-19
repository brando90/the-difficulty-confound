#!/bin/bash
# Experiment 13 (13-GPU arm) on Sherlock: one array task = one grid shard.
# Submit from $SCRATCH/tdc_exp13 (never compute on a login node):
#   cd $SCRATCH/tdc_exp13 && mkdir -p slurm_logs
#   sbatch --export=ALL,TDC_COMMIT=<sha> the-difficulty-confound/experiments/13_ginc_hmm_complexity_vs_alignment/launch/sherlock.sbatch
# MODE=run (default) trains shard $SLURM_ARRAY_TASK_ID (0..359) unless it is DONE.
# MODE=verify replays a DONE shard with verify.py and writes VERIFIED.
# MODE=pilot runs GPU pilot job $SLURM_ARRAY_TASK_ID; submit with --array=0-8.
#SBATCH --job-name=tdc_exp13
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-359%16
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
# Optional: only 80 GB GPUs (list features with: sh_node_feat -p gpu | grep GPU_SKU)
##SBATCH -C GPU_MEM:80GB
set -euo pipefail

die() { echo "sherlock.sbatch: ERROR: $*" >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || die "submit this with sbatch; never run it on a login node"
ID=${SLURM_ARRAY_TASK_ID:?submit as an array job (--array)}
MODE=${MODE:-run}
EXP_NAME=exp13
EXP_DIR=experiments/13_ginc_hmm_complexity_vs_alignment
ARM=gpu
NSHARDS=360
NPILOT=9
JOB_MIN_GB=${JOB_MIN_GB:-10}

: "${SCRATCH:?SCRATCH is not set}"
BASE=${BASE:-$SCRATCH/tdc_$EXP_NAME}
REPO=$BASE/the-difficulty-confound
OUT=${OUT:-$BASE/runs/gpu_v1}
CODE=$REPO/$EXP_DIR/expt_v1

case $MODE in
  run) LIMIT=$NSHARDS MARK=$OUT/shards CLAIMS=$OUT/claims TAG=shard ;;
  verify) LIMIT=$NSHARDS MARK=$OUT/shards CLAIMS=$OUT/claims_verify TAG=verify ;;
  pilot) LIMIT=$NPILOT MARK=$OUT/pilot CLAIMS=$OUT/claims_pilot TAG=pilot ;;
  *) die "MODE must be run, verify or pilot (got '$MODE')" ;;
esac
((ID < LIMIT)) || die "array index $ID is outside 0..$((LIMIT - 1)) for MODE=$MODE"
PADDED=$(printf '%05d' "$ID")
mkdir -p "$OUT/logs" "$MARK" "$CLAIMS"

# ---- repository at the frozen commit -----------------------------------------
: "${TDC_COMMIT:?pass TDC_COMMIT=<sha> with --export=ALL,TDC_COMMIT=<sha>}"
WANT=$(git -C "$REPO" rev-parse --verify "${TDC_COMMIT}^{commit}") || die "TDC_COMMIT not found in $REPO"
[[ $(git -C "$REPO" rev-parse HEAD) == "$WANT" ]] || die "HEAD of $REPO is not $WANT; check out the frozen commit first"
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || die "tracked files are modified in $REPO"

# ---- skip finished work (resumable) --------------------------------------------
if [[ $MODE == verify ]]; then
  [[ -f $MARK/$PADDED/DONE ]] || die "shard $ID is not DONE; nothing to verify"
  if [[ -f $MARK/$PADDED/VERIFIED ]]; then echo "shard $ID already VERIFIED; exiting"; exit 0; fi
elif [[ -f $MARK/$PADDED/DONE ]]; then
  echo "$TAG $ID already DONE; exiting"
  exit 0
fi
if ! mkdir "$CLAIMS/$PADDED" 2>/dev/null; then
  die "$TAG $ID is claimed by $(cat "$CLAIMS/$PADDED/owner" 2>/dev/null || echo unknown) but not DONE. If that job is dead: move the partial folder aside, rm -r $CLAIMS/$PADDED, log it in CKPT, resubmit this index. No automatic retries."
fi
echo "sherlock job $SLURM_JOB_ID array $ID on $(hostname -s) $(date '+%m-%d-%Y %H:%M:%S')" >"$CLAIMS/$PADDED/owner"

# ---- disk headroom ------------------------------------------------------------------
AVAIL_GB=$(df -P -BG "$OUT" | awk 'NR == 2 { gsub("G", "", $4); print $4 }')
((AVAIL_GB >= JOB_MIN_GB)) || die "only ${AVAIL_GB} GB free under $OUT (need $JOB_MIN_GB GB for one shard)"

# ---- Python module and virtual environment ----------------------------------------
if ! type module >/dev/null 2>&1; then
  set +u
  for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f $init ]]; then source "$init"; break; fi
  done
  set -u
fi
set +u
module load "${PY_MODULE:-python/3.12}" || die "module load ${PY_MODULE:-python/3.12} failed; check 'module avail python' and set PY_MODULE"
set -u
if [[ -n ${GROUP_HOME:-} ]]; then VENV_ROOT=$GROUP_HOME/$USER/venvs; else VENV_ROOT=$HOME/venvs; fi
VENV=${VENV:-$VENV_ROOT/tdc_$EXP_NAME}
TORCH_VERSION=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["gpu_torch_version"])' "$CODE/config.json")
TORCH_VERSION=${TORCH_VERSION%%+*}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
if [[ ! -f $VENV/.ready ]]; then
  mkdir -p "$VENV_ROOT"
  if mkdir "$VENV.lock" 2>/dev/null; then
    echo "building $VENV with torch==$TORCH_VERSION from $TORCH_INDEX"
    if python3 -m venv "$VENV" &&
      "$VENV/bin/pip" install --upgrade pip &&
      "$VENV/bin/pip" install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX" &&
      "$VENV/bin/pip" install -r "$CODE/requirements-gpu.txt" &&
      "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"'; then
      echo "torch==$TORCH_VERSION $TORCH_INDEX $(date +%m-%d-%Y)" >"$VENV/.ready"
      rmdir "$VENV.lock"
    else
      rmdir "$VENV.lock"
      rm -r "$CLAIMS/$PADDED"
      die "virtual environment build failed; see this job's .err file"
    fi
  else
    echo "another task is building $VENV; waiting (up to 30 min)"
    for _ in $(seq 1 60); do
      [[ -f $VENV/.ready ]] && break
      sleep 30
    done
    if [[ ! -f $VENV/.ready ]]; then
      rm -r "$CLAIMS/$PADDED"
      die "$VENV not ready after 30 min (stale $VENV.lock?)"
    fi
  fi
fi
PY=$VENV/bin/python

# ---- GPU check -----------------------------------------------------------------------
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
if ! "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0) else 1)'; then
  rm -r "$CLAIMS/$PADDED"
  die "CUDA unavailable or GPU compute capability < 8.0 (bfloat16 needs >= 8.0); add a GPU constraint (sh_node_feat -p gpu) and resubmit"
fi

# ---- run --------------------------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_GPU:-8}
cd "$REPO"
LOG=$OUT/logs/${TAG}_$PADDED.log
START=$(date +%s)
set +e
case $MODE in
  run) "$PY" -u "$CODE/run.py" --arm "$ARM" --shard "$ID" --output "$OUT" >>"$LOG" 2>&1 ;;
  verify) "$PY" -u "$CODE/verify.py" --arm "$ARM" --shard "$ID" --output "$OUT" >>"$LOG" 2>&1 ;;
  pilot) "$PY" -u "$CODE/run.py" --arm "$ARM" --pilot --pilot-job "$ID" --output "$OUT/pilot" >>"$LOG" 2>&1 ;;
esac
RC=$?
set -e
printf '%s\t%s\t%s\tsherlock:%s\t%s\t%s\n' "$(date '+%m-%d-%Y %H:%M:%S')" "$TAG" "$PADDED" "$SLURM_JOB_ID" "$RC" "$(($(date +%s) - START))" >>"$OUT/logs/status_sherlock.tsv"
echo "$TAG $ID finished with exit code $RC; log: $LOG"
exit "$RC"
