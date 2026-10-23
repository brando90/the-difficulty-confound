#!/usr/bin/env bash
# Experiment 13 (13-GPU arm): run a range of grid shards on ONE SNAP GPU node,
# one process per free GPU, detached in the tmux session tdc_exp13.
#
# Run this ON the node, after logging in once (it never opens SSH itself):
#   cd /dfs/scratch0/$USER/tdc_exp13/the-difficulty-confound
#   TDC_COMMIT=<sha> bash experiments/13_ginc_hmm_complexity_vs_alignment/launch/snap.sh <host> <first> <last>
#
# MODE=run (default)  train shards <first>..<last> (0..359), skipping DONE ones
# MODE=verify         run verify.py on DONE shards, writing VERIFIED markers
# MODE=pilot          run GPU pilot jobs <first>..<last> (0..8)
# Optional environment: OUT, GPUS (comma list), MAX_GPUS, MAX_USED_MB,
# MAX_UTIL, MIN_FREE_MB, REQUIRED_GB, VENV, PYTHON, TORCH_INDEX,
# ALLOW_OTHER_REPO_PATH=1.
set -euo pipefail

# ---- experiment constants -------------------------------------------------
EXP_NAME=exp13
EXP_DIR=experiments/13_ginc_hmm_complexity_vs_alignment
ARM=gpu
SESSION=tdc_exp13
NSHARDS=360
NPILOT=9
DEFAULT_OUT=/dfs/scratch0/$USER/tdc_exp13/runs/gpu_v1
EXPECTED_REPO=/dfs/scratch0/$USER/tdc_exp13/the-difficulty-confound
# ---------------------------------------------------------------------------

MODE=${MODE:-run}
OUT=${OUT:-$DEFAULT_OUT}
MAX_GPUS=${MAX_GPUS:-8}
MAX_USED_MB=${MAX_USED_MB:-2000}
MAX_UTIL=${MAX_UTIL:-10}
MIN_FREE_MB=${MIN_FREE_MB:-20000}
if [[ $MODE == pilot ]]; then REQUIRED_GB=${REQUIRED_GB:-20}; else REQUIRED_GB=${REQUIRED_GB:-300}; fi
HOME_MIN_GB=${HOME_MIN_GB:-20}
VENV=${VENV:-$HOME/venvs/tdc_$EXP_NAME}
PYTHON=${PYTHON:-python3}
THREADS_PER_GPU=${THREADS_PER_GPU:-8}

die() { echo "snap.sh: ERROR: $*" >&2; exit 1; }
say() { echo "snap.sh: $*"; }

# ---- 1. arguments -----------------------------------------------------------
[[ $# -eq 3 ]] || die "usage: [MODE=run|verify|pilot] TDC_COMMIT=<sha> bash $0 <host> <first> <last>"
HOST=$1
FIRST=$2
LAST=$3
[[ $FIRST =~ ^[0-9]+$ && $LAST =~ ^[0-9]+$ ]] || die "first and last must be non-negative integers"
case $MODE in
  run | verify) LIMIT=$NSHARDS ;;
  pilot) LIMIT=$NPILOT ;;
  *) die "MODE must be run, verify or pilot (got '$MODE')" ;;
esac
FIRST=$((10#$FIRST))
LAST=$((10#$LAST))
((FIRST <= LAST)) || die "first ($FIRST) > last ($LAST)"
((LAST < LIMIT)) || die "last ($LAST) must be < $LIMIT for MODE=$MODE"

# ---- 2. right machine -------------------------------------------------------
THIS_HOST=$(hostname -s)
[[ $THIS_HOST == "$HOST" ]] || die "this is '$THIS_HOST', not '$HOST'. Log in to $HOST once (no retry loops) and rerun there."

# ---- 3. repository at the frozen commit --------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
if [[ $REPO != "$EXPECTED_REPO" && ${ALLOW_OTHER_REPO_PATH:-0} != 1 ]]; then
  die "repository is at $REPO; expected the shared DFS clone $EXPECTED_REPO (set ALLOW_OTHER_REPO_PATH=1 to override)"
fi
: "${TDC_COMMIT:?set TDC_COMMIT to the commit named in CKPT (freeze SHA for MODE=pilot, pilot-disclosure SHA for run/verify)}"
WANT=$(git -C "$REPO" rev-parse --verify "${TDC_COMMIT}^{commit}") || die "TDC_COMMIT=$TDC_COMMIT is not a commit in $REPO (git fetch first)"
HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
[[ $HEAD_SHA == "$WANT" ]] || die "HEAD is $HEAD_SHA but TDC_COMMIT is $WANT; run: git -C $REPO checkout --detach $WANT"
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || die "tracked files are modified in $REPO; refusing to run"
CODE=$REPO/$EXP_DIR/expt_v1
for f in run.py verify.py config.json requirements-gpu.txt; do
  [[ -f $CODE/$f ]] || die "missing $CODE/$f"
done

# ---- 4. free GPUs -------------------------------------------------------------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; is $HOST a GPU node?"
FREE_GPUS=()
while IFS=',' read -r idx used free util; do
  idx=${idx//[[:space:]]/}
  used=${used//[[:space:]]/}
  free=${free//[[:space:]]/}
  util=${util//[[:space:]]/}
  [[ $idx =~ ^[0-9]+$ && $used =~ ^[0-9]+$ && $free =~ ^[0-9]+$ && $util =~ ^[0-9]+$ ]] || continue
  if [[ -n ${GPUS:-} && ",${GPUS}," != *",${idx},"* ]]; then continue; fi
  if ((used <= MAX_USED_MB && util <= MAX_UTIL && free >= MIN_FREE_MB)); then
    FREE_GPUS+=("$idx")
  else
    say "GPU $idx busy or too small (used ${used} MB, free ${free} MB, util ${util}%), skipping"
  fi
done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
((${#FREE_GPUS[@]} > 0)) || die "no free GPU on $HOST (thresholds: used <= $MAX_USED_MB MB, util <= $MAX_UTIL%, free >= $MIN_FREE_MB MB)"
FREE_GPUS=("${FREE_GPUS[@]:0:MAX_GPUS}")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
say "using GPUs ${FREE_GPUS[*]} ($GPU_NAME)"

# ---- 5. disk headroom ---------------------------------------------------------
mkdir -p "$OUT/logs" "$OUT/claims" "$OUT/claims_verify" "$OUT/claims_pilot" "$OUT/shards" "$OUT/pilot"
avail_gb() { df -P -BG "$1" | awk 'NR == 2 { gsub("G", "", $4); print $4 }'; }
OUT_GB=$(avail_gb "$OUT")
HOME_GB=$(avail_gb "$HOME")
((OUT_GB >= REQUIRED_GB)) || die "only ${OUT_GB} GB free under $OUT; need $REQUIRED_GB GB (set REQUIRED_GB lower only if most shards are already DONE)"
((HOME_GB >= HOME_MIN_GB)) || die "only ${HOME_GB} GB free in $HOME; need $HOME_MIN_GB GB for the virtual environment"

# ---- 6. tmux --------------------------------------------------------------------
command -v tmux >/dev/null || die "tmux not found"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  die "tmux session $SESSION already exists on $HOST; attach with 'tmux attach -t $SESSION' or stop it first"
fi

# ---- 7. virtual environment on node-local disk ---------------------------------
CONFIG=$CODE/config.json
TORCH_VERSION=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["gpu_torch_version"])' "$CONFIG")
TORCH_VERSION=${TORCH_VERSION%%+*}
if [[ -z ${TORCH_INDEX:-} ]]; then
  DRIVER_CUDA=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | head -n 1 | awk '{ print $3 }')
  DRIVER_CUDA=${DRIVER_CUDA:-0.0}
  CUDA_MAJOR=${DRIVER_CUDA%%.*}
  CUDA_MINOR=${DRIVER_CUDA#*.}
  # torch 2.6.0 (the pinned GPU version, shared with Sherlock) has cu124 and cu126 wheels, none for B200 (sm_100).
  [[ $GPU_NAME != *B200* ]] || die "torch $TORCH_VERSION has no B200 support; use skampere1 (A100) or skampere2 (H200)"
  if ((CUDA_MAJOR > 12 || (CUDA_MAJOR == 12 && CUDA_MINOR >= 6))); then
    TORCH_INDEX=https://download.pytorch.org/whl/cu126
  elif ((CUDA_MAJOR == 12 && CUDA_MINOR >= 4)); then
    TORCH_INDEX=https://download.pytorch.org/whl/cu124
  else
    die "driver reports CUDA $DRIVER_CUDA; set TORCH_INDEX to a matching https://download.pytorch.org/whl/cuXXX index"
  fi
fi
if [[ ! -f $VENV/.ready ]]; then
  say "building $VENV with torch==$TORCH_VERSION from $TORCH_INDEX"
  mkdir -p "$(dirname "$VENV")"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
  "$VENV/bin/pip" install --only-binary=:all: -r "$CODE/requirements-gpu.txt"
  "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  echo "torch==$TORCH_VERSION $TORCH_INDEX $(date +%m-%d-%Y)" >"$VENV/.ready"
fi
PY=$VENV/bin/python
"$PY" -c 'import torch, sys; ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0); sys.exit(0 if ok else 1)' ||
  die "CUDA unavailable in $VENV or GPU compute capability < 8.0 (bfloat16 needs >= 8.0)"

# ---- 8. preflight: --check-only builds expt_v1/prep/gpu once (about 1 h); later calls reuse it ------
if [[ -f $CODE/prep/gpu/tasks.json && -f $CODE/prep/gpu/mixture.npz ]]; then
  say "concepts and references present in $CODE/prep/gpu (built by an earlier --check-only); not rebuilding"
else
  say "running run.py --arm $ARM --check-only (up to 3 h; builds $CODE/prep/gpu)"
  (cd "$REPO" && OMP_NUM_THREADS=1 timeout 10800 "$PY" -u "$CODE/run.py" --arm "$ARM" --check-only --workers "$THREADS_PER_GPU") \
    >"$OUT/logs/check_only_$HOST.log" 2>&1 || die "--check-only failed; see $OUT/logs/check_only_$HOST.log"
fi

# ---- 9. pending work ----------------------------------------------------------------
case $MODE in
  run) MARK_ROOT=$OUT/shards CLAIMS=$OUT/claims TAG=shard ;;
  verify) MARK_ROOT=$OUT/shards CLAIMS=$OUT/claims_verify TAG=verify ;;
  pilot) MARK_ROOT=$OUT/pilot CLAIMS=$OUT/claims_pilot TAG=pilot ;;
esac
PENDING=()
for ((s = FIRST; s <= LAST; s++)); do
  d=$MARK_ROOT/$(printf '%05d' "$s")
  if [[ $MODE == verify ]]; then
    if [[ ! -f $d/DONE ]]; then
      say "shard $s is not DONE, cannot verify it yet"
      continue
    fi
    if [[ -f $d/VERIFIED ]]; then continue; fi
  elif [[ -f $d/DONE ]]; then
    continue
  fi
  if [[ -d $CLAIMS/$(printf '%05d' "$s") ]]; then
    say "id $s is claimed but not finished (running elsewhere, or crashed: see launch/README.md); skipping"
    continue
  fi
  PENDING+=("$s")
done
((${#PENDING[@]} > 0)) || { say "nothing to do for ids $FIRST..$LAST in MODE=$MODE"; exit 0; }
say "${#PENDING[@]} ids pending in MODE=$MODE"

# ---- 10. one worker script per GPU, round-robin assignment ----------------------------
NG=${#FREE_GPUS[@]}
WORKERS=()
for ((k = 0; k < NG; k++)); do
  g=${FREE_GPUS[$k]}
  MINE=()
  for ((j = k; j < ${#PENDING[@]}; j += NG)); do MINE+=("${PENDING[$j]}"); done
  ((${#MINE[@]} > 0)) || continue
  W=$OUT/logs/worker_${HOST}_gpu${g}_${MODE}.sh
  {
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$g"
    printf 'export OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q\n' "$THREADS_PER_GPU" "$THREADS_PER_GPU"
    printf 'PY=%q\nCODE=%q\nREPO=%q\nOUT=%q\nCLAIMS=%q\nTAG=%q\nMODE=%q\nARM=%q\nHOST=%q\n' \
      "$PY" "$CODE" "$REPO" "$OUT" "$CLAIMS" "$TAG" "$MODE" "$ARM" "$HOST"
    printf 'IDS=(%s)\n' "${MINE[*]}"
    cat <<'BODY'
cd "$REPO" || exit 1
for s in "${IDS[@]}"; do
  id=$(printf '%05d' "$s")
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    echo "$(date '+%m-%d-%Y %H:%M:%S') skip $TAG $id: already claimed"
    continue
  fi
  echo "$HOST gpu$CUDA_VISIBLE_DEVICES $(date '+%m-%d-%Y %H:%M:%S')" >"$CLAIMS/$id/owner"
  start=$(date +%s)
  echo "$(date '+%m-%d-%Y %H:%M:%S') start $TAG $id on gpu$CUDA_VISIBLE_DEVICES"
  case $MODE in
    run) "$PY" -u "$CODE/run.py" --arm "$ARM" --shard "$s" --output "$OUT" >>"$OUT/logs/${TAG}_$id.log" 2>&1 ;;
    verify) "$PY" -u "$CODE/verify.py" --arm "$ARM" --shard "$s" --output "$OUT" >>"$OUT/logs/${TAG}_$id.log" 2>&1 ;;
    pilot) "$PY" -u "$CODE/run.py" --arm "$ARM" --pilot --pilot-job "$s" --output "$OUT/pilot" >>"$OUT/logs/${TAG}_$id.log" 2>&1 ;;
  esac
  rc=$?
  printf '%s\t%s\t%s\tgpu%s\t%s\t%s\n' "$(date '+%m-%d-%Y %H:%M:%S')" "$TAG" "$id" "$CUDA_VISIBLE_DEVICES" "$rc" "$(($(date +%s) - start))" >>"$OUT/logs/status_$HOST.tsv"
  echo "$(date '+%m-%d-%Y %H:%M:%S') end $TAG $id exit $rc"
done
echo "worker finished; no automatic retries (see launch/README.md for failed ids)"
BODY
  } >"$W"
  chmod +x "$W"
  WORKERS+=("$g:$W")
done

# ---- 11. launch in tmux ------------------------------------------------------------------
FIRST_WINDOW=1
for gw in "${WORKERS[@]}"; do
  g=${gw%%:*}
  W=${gw#*:}
  CMD="bash $(printf '%q' "$W") 2>&1 | tee -a $(printf '%q' "$OUT/logs/worker_${HOST}_gpu${g}_${MODE}.log")"
  if ((FIRST_WINDOW)); then
    tmux new-session -d -s "$SESSION" -n "gpu$g" "$CMD"
    tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
    FIRST_WINDOW=0
  else
    tmux new-window -t "$SESSION" -n "gpu$g" "$CMD"
  fi
done

say "launched ${#WORKERS[@]} workers in tmux session $SESSION on $HOST (MODE=$MODE)"
cat <<EOF
Watch:    tmux attach -t $SESSION        (detach: Ctrl-b d)
          tail -f $OUT/logs/worker_${HOST}_gpu*_${MODE}.log
          ls $MARK_ROOT/*/DONE 2>/dev/null | wc -l       (finished ids)
          cat $OUT/logs/status_$HOST.tsv                  (exit codes; nonzero = failed)
Stop:     tmux kill-session -t $SESSION
          Then, for each id claimed but not DONE: move its partial folder aside,
          remove its claim with rm -r $CLAIMS/<id>, and log it in CKPT_ginc_hmm.md.
EOF
