#!/usr/bin/env bash
# 13-GPU on skampere1: repair the interrupted checkout, build the venv on DFS, run the unit tests, then snap.sh MODE=pilot
# (it builds prep/gpu with --check-only and starts the pilot workers in tmux session tdc_exp13) on GPUs 3 and 5 only.
set -euo pipefail
B=/dfs/scratch0/$USER/tdc_exp13
REPO=$B/the-difficulty-confound
SHA=2c625111
export VENV=/lfs/skampere1/0/laoluog1/venvs/tdc_exp13
export PIP_CACHE_DIR=/lfs/skampere1/0/laoluog1/pip_cache
CODE=$REPO/experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || { echo "checkout still dirty"; exit 1; }
echo "$(date '+%m-%d-%Y %H:%M:%S') checkout clean at $(git -C "$REPO" rev-parse --short HEAD)"
if [[ ! -f $VENV/.ready ]]; then
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126
  "$VENV/bin/pip" install -q --only-binary=:all: -r "$CODE/requirements-gpu.txt"
  "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available()'
  echo "torch==2.6.0 https://download.pytorch.org/whl/cu126 $(date +%m-%d-%Y)" >"$VENV/.ready"
fi
"$VENV/bin/python" -c 'import torch, numpy, scipy; print("torch", torch.__version__, "cuda", torch.version.cuda, "numpy", numpy.__version__, "scipy", scipy.__version__)'
echo "$(date '+%m-%d-%Y %H:%M:%S') unit tests"
(cd "$CODE" && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$VENV/bin/python" -m pytest -q . 2>&1 | tail -n 2)
cd "$REPO"
echo "$(date '+%m-%d-%Y %H:%M:%S') snap.sh MODE=pilot (check-only builds prep first)"
MODE=pilot TDC_COMMIT=$SHA GPUS=3,5 MAX_GPUS=2 MAX_USED_MB=80000 MAX_UTIL=100 MIN_FREE_MB=24000 \
  bash experiments/13_ginc_hmm_complexity_vs_alignment/launch/snap.sh skampere1 0 8
echo "$(date '+%m-%d-%Y %H:%M:%S') SETUP_DONE"
