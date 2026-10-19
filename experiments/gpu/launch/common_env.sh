#!/usr/bin/env bash
# Sourced by the launchers: Python module (Sherlock), virtual environment with CUDA torch, cache folders.
# Needs: BASE (work folder), REPO (clone at the frozen commit). Sets: PY, DATA, OUT, HF_HOME.
set -euo pipefail
GPU_DIR=$REPO/experiments/gpu
DATA=${DATA:-$BASE/data}
OUT=${OUT:-$BASE/runs}
export HF_HOME=${HF_HOME:-$BASE/hf_cache}
export TOKENIZERS_PARALLELISM=false
if type module >/dev/null 2>&1 || [[ -f /etc/profile.d/lmod.sh ]]; then
  set +u
  type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
  module load "${PY_MODULE:-python/3.12}" || { echo "module load ${PY_MODULE:-python/3.12} failed; check 'module avail python' and set PY_MODULE" >&2; exit 2; }
  set -u
fi
VENV=${VENV:-$BASE/venv}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
if [[ ! -f $VENV/.ready ]]; then
  if mkdir "$VENV.lock" 2>/dev/null; then
    echo "building $VENV (torch 2.14.1 from $TORCH_INDEX)"
    python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip &&
      "$VENV/bin/pip" install -q "torch==2.14.1" --index-url "$TORCH_INDEX" &&
      "$VENV/bin/pip" install -q -r "$GPU_DIR/common/requirements-gpu.txt" &&
      echo "torch==2.14.1 $TORCH_INDEX $(date +%m-%d-%Y)" >"$VENV/.ready"
    rmdir "$VENV.lock"
    [[ -f $VENV/.ready ]] || { echo "virtual environment build failed" >&2; exit 2; }
  else
    echo "another job is building $VENV; waiting up to 30 min"
    for _ in $(seq 1 60); do [[ -f $VENV/.ready ]] && break; sleep 30; done
    [[ -f $VENV/.ready ]] || { echo "$VENV not ready after 30 min (stale $VENV.lock?)" >&2; exit 2; }
  fi
fi
PY=$VENV/bin/python
check_commit() {
  : "${TDC_COMMIT:?pass TDC_COMMIT=<sha> (the freeze commit for the pilot, the pilot-disclosure commit afterwards; see CKPT.md)}"
  local want
  want=$(git -C "$REPO" rev-parse --verify "${TDC_COMMIT}^{commit}") || { echo "TDC_COMMIT not found in $REPO (git fetch first)" >&2; exit 2; }
  [[ $(git -C "$REPO" rev-parse HEAD) == "$want" ]] || { echo "HEAD of $REPO is not $want; git -C $REPO checkout --detach $want" >&2; exit 2; }
  [[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || { echo "tracked files are modified in $REPO; refusing to run" >&2; exit 2; }
}
