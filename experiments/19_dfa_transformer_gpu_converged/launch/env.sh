#!/usr/bin/env bash
# Sourced by the Experiment 19 jobs. Uses the virtual environment AT-441's prep job builds at $BASE/venv
# (torch 2.14.1 CUDA, numpy, scipy); builds it the same way if it is missing. Sets PY and OUT.
set -euo pipefail
if type module >/dev/null 2>&1 || [[ -f /etc/profile.d/lmod.sh ]]; then
  set +u
  type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
  module load "${PY_MODULE:-python/3.12}" || { echo "module load ${PY_MODULE:-python/3.12} failed" >&2; exit 2; }
  set -u
fi
VENV=${VENV:-$BASE/venv}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
if [[ ! -f $VENV/.ready ]]; then
  if mkdir "$VENV.lock" 2>/dev/null; then
    echo "building $VENV"
    python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip &&
      "$VENV/bin/pip" install -q "torch==2.14.1" --index-url "$TORCH_INDEX" &&
      "$VENV/bin/pip" install -q numpy==2.5.4 scipy==1.18.1 &&
      echo "torch==2.14.1 $TORCH_INDEX $(date +%m-%d-%Y) (exp19)" >"$VENV/.ready"
    rmdir "$VENV.lock"
    [[ -f $VENV/.ready ]] || { echo "virtual environment build failed" >&2; exit 2; }
  else
    echo "another job is building $VENV; waiting up to 30 min"
    for _ in $(seq 1 60); do [[ -f $VENV/.ready ]] && break; sleep 30; done
    [[ -f $VENV/.ready ]] || { echo "$VENV not ready after 30 min" >&2; exit 2; }
  fi
fi
PY=$VENV/bin/python
OUT=${OUT:-$BASE/exp19_results}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
