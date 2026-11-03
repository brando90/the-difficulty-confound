#!/bin/bash
# Experiment 13 (13-GPU arm) on Sherlock: build the concepts and exact references once, on a CPU node, before any
# GPU job (run.py --arm gpu --check-only into expt_v1/prep/gpu; about 1 h). Submit from $SCRATCH/tdc_exp13:
#   cd $SCRATCH/tdc_exp13 && mkdir -p slurm_logs
#   sbatch --export=ALL,TDC_COMMIT=<sha> the-difficulty-confound/experiments/13_ginc_hmm_complexity_vs_alignment/launch/sherlock_prep.sbatch
# Then chain the GPU pilot with --dependency=afterok:<this job id>. It also builds the shared virtual environment
# (torch from the CUDA index; CUDA itself is checked by the GPU jobs).
#SBATCH --job-name=tdc_exp13_prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com,laoluoguneye@gmail.com
set -euo pipefail

die() { echo "sherlock_prep.sbatch: ERROR: $*" >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || die "submit this with sbatch; never run it on a login node"
EXP_NAME=exp13
EXP_DIR=experiments/13_ginc_hmm_complexity_vs_alignment
: "${SCRATCH:?SCRATCH is not set}"
BASE=${BASE:-$SCRATCH/tdc_$EXP_NAME}
REPO=$BASE/the-difficulty-confound
CODE=$REPO/$EXP_DIR/expt_v1
OUT=${OUT:-$BASE/runs/gpu_v1}
mkdir -p "$OUT/logs"

if ! type module >/dev/null 2>&1; then
  set +u
  for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f $init ]]; then source "$init"; break; fi
  done
  set -u
fi
set +u
module load system git 2>/dev/null || echo "note: 'module load system git' failed; using $(git --version)"
module load "${PY_MODULE:-python/3.12}" || die "module load ${PY_MODULE:-python/3.12} failed"
set -u

: "${TDC_COMMIT:?pass TDC_COMMIT=<sha> with --export=ALL,TDC_COMMIT=<sha>}"
WANT=$(git -C "$REPO" rev-parse --verify "${TDC_COMMIT}^{commit}") || die "TDC_COMMIT not found in $REPO"
[[ $(git -C "$REPO" rev-parse HEAD) == "$WANT" ]] || die "HEAD of $REPO is not $WANT; check out the commit first"
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || die "tracked files are modified in $REPO"

if [[ -n ${GROUP_HOME:-} ]]; then VENV_ROOT=$GROUP_HOME/$USER/venvs; else VENV_ROOT=$HOME/venvs; fi
VENV=${VENV:-$VENV_ROOT/tdc_$EXP_NAME}
TORCH_VERSION=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["gpu_torch_version"])' "$CODE/config.json")
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
if [[ ! -f $VENV/.ready ]]; then
  mkdir -p "$VENV_ROOT"
  mkdir "$VENV.lock" 2>/dev/null || die "$VENV.lock exists: another job is building $VENV (or a stale lock)"
  echo "building $VENV with torch==$TORCH_VERSION from $TORCH_INDEX"
  if python3 -m venv "$VENV" &&
    "$VENV/bin/pip" install --upgrade pip &&
    "$VENV/bin/pip" install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX" &&
    "$VENV/bin/pip" install --only-binary=:all: -r "$CODE/requirements-gpu.txt"; then
    echo "torch==$TORCH_VERSION $TORCH_INDEX $(date +%m-%d-%Y)" >"$VENV/.ready"
    rmdir "$VENV.lock"
  else
    rmdir "$VENV.lock"
    die "virtual environment build failed; see this job's .err file"
  fi
fi
PY=$VENV/bin/python

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
START=$(date +%s)
"$PY" -u "$CODE/run.py" --arm gpu --check-only --workers "${SLURM_CPUS_PER_TASK:-8}" 2>&1 | tee -a "$OUT/logs/check_only_sherlock.log"
"$PY" -m pytest -q "$CODE" 2>&1 | tail -n 3 | tee -a "$OUT/logs/check_only_sherlock.log"
echo "prep finished in $(($(date +%s) - START)) s; prep in $CODE/prep/gpu, receipt in $CODE/results_check_gpu.json"
