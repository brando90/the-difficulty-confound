#!/usr/bin/env bash
# 13-GPU extra shard workers on skampere2: venv on node-local disk, then one shard_worker2.sh per GPU with >= 24 GB free
# (least busy first, at most 8) in tmux session tdc_exp13_shards2. Same DFS repo (at the disclosure commit), prep,
# output and claims as skampere1, so no shard runs twice.
set -euo pipefail
U=laoluog1
B=/dfs/scratch0/$U/tdc_exp13
REPO=$B/the-difficulty-confound
CODE=$REPO/experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1
VENV=/lfs/skampere2/0/$U/venvs/tdc_exp13
OUT=/dfs/scratch1/$U/tdc_exp13/runs/gpu_v1
export PIP_CACHE_DIR=/lfs/skampere2/0/$U/pip_cache
[[ $(git -C "$REPO" rev-parse HEAD) == $(cat "$B/state/DISCLOSURE_SHA") ]] || { echo "repo not at the disclosure commit"; exit 1; }
if [[ ! -f $VENV/.ready ]]; then
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126
  "$VENV/bin/pip" install -q --only-binary=:all: -r "$CODE/requirements-gpu.txt"
  "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available()'
  echo "torch==2.6.0 https://download.pytorch.org/whl/cu126 $(date +%m-%d-%Y)" >"$VENV/.ready"
fi
"$VENV/bin/python" -c 'import torch, numpy, scipy; print("torch", torch.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__)'
GPUS=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
  awk -F', ' '$2 >= 24000 { print $3, $1 }' | sort -n | awk 'NR <= 8 { print $2 }')
echo "$(date '+%m-%d-%Y %H:%M:%S') skampere2 workers on GPUs" $GPUS
first=1
for g in $GPUS; do
  CMD="bash $B/shard_worker2.sh $g $OUT 2>&1 | tee -a $OUT/logs/worker_skampere2_gpu$g.log"
  if ((first)); then tmux new-session -d -s tdc_exp13_shards2 -n "gpu$g" "$CMD"; tmux set-option -t tdc_exp13_shards2 remain-on-exit on >/dev/null; first=0
  else tmux new-window -t tdc_exp13_shards2 -n "gpu$g" "$CMD"; fi
done
echo "$(date '+%m-%d-%Y %H:%M:%S') S2_LAUNCHED"
