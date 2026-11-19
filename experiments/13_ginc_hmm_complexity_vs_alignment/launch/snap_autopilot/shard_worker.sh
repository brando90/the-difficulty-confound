#!/usr/bin/env bash
# One 13-GPU shard worker on one shared GPU of skampere1 (started by autopilot.sh). Walks shards L, M, S (largest first),
# claims each by mkdir, waits while the GPU has < MIN_FREE_MB free, trains it, then verifies it on the same GPU.
# Stops before the next shard if $STATE/STOP or $STATE/HALT exists. A failed shard keeps its claim (manual rerun only).
set -uo pipefail
G=$1
OUT=$2
U=laoluog1
B=/dfs/scratch0/$U/tdc_exp13
REPO=$B/the-difficulty-confound
CODE=$REPO/experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1
PY=/lfs/skampere1/0/laoluog1/venvs/tdc_exp13/bin/python
STATE=$B/state
MIN_FREE_MB=${MIN_FREE_MB:-24000}
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/cache \
  CUDA_CACHE_PATH=$B/cache/nv TORCH_HOME=$B/cache/torch MPLCONFIGDIR=$B/cache/mpl
ts() { date '+%m-%d-%Y %H:%M:%S'; }
cd "$REPO" || exit 1
for s in $(seq 240 359) $(seq 120 239) $(seq 0 119); do
  [[ -f $STATE/STOP || -f $STATE/HALT ]] && { echo "$(ts) stop requested; worker gpu$G exits"; break; }
  id=$(printf '%05d' "$s")
  [[ -f $OUT/shards/$id/DONE ]] && continue
  mkdir "$OUT/claims/$id" 2>/dev/null || continue
  echo "skampere1 gpu$G $(ts)" >"$OUT/claims/$id/owner"
  while true; do
    free=$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    ((free >= MIN_FREE_MB)) && break
    echo "$(ts) gpu$G has ${free} MB free (< $MIN_FREE_MB); waiting before shard $id"
    sleep 300
  done
  echo "$(ts) start shard $id on gpu$G (${free} MB free)"
  start=$(date +%s)
  "$PY" -u "$CODE/run.py" --arm gpu --shard "$s" --output "$OUT" >>"$OUT/logs/shard_$id.log" 2>&1
  rc=$?
  vrc=-
  if ((rc == 0)); then
    "$PY" -u "$CODE/verify.py" --arm gpu --shard "$s" --output "$OUT" >>"$OUT/logs/verify_$id.log" 2>&1
    vrc=$?
  fi
  printf '%s\t%s\t%s\tgpu%s\t%s\t%s\t%s\n' "$(ts)" shard "$id" "$G" "$rc" "$vrc" "$(($(date +%s) - start))" >>"$OUT/logs/status_skampere1.tsv"
  echo "$(ts) end shard $id: train exit $rc, verify exit $vrc, $(($(date +%s) - start)) s"
done
echo "$(ts) worker gpu$G finished"
