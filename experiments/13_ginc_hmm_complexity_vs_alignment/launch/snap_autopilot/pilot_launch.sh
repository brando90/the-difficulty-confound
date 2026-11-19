#!/usr/bin/env bash
# 13-GPU pilot on skampere1 without snap.sh (its nvidia-smi | head pipes exit 141 under pipefail): check-only builds
# prep/gpu, then one pilot worker per shared GPU (>= MIN_FREE_MB free, least busy first, at most 3) in tmux session
# tdc_exp13. Jobs L first; claims by mkdir; status lines in the snap.sh format (TAG pilot) for autopilot.sh.
set -uo pipefail
U=laoluog1
B=/dfs/scratch0/$U/tdc_exp13
REPO=$B/the-difficulty-confound
CODE=$REPO/experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1
PY=/lfs/skampere1/0/$U/venvs/tdc_exp13/bin/python
OUT=$B/runs/gpu_v1
MIN_FREE_MB=${MIN_FREE_MB:-24000}
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv TORCH_HOME=$B/cache/torch MPLCONFIGDIR=$B/cache/mpl
mkdir -p "$OUT/logs" "$OUT/pilot" "$OUT/claims_pilot"
cd "$REPO" || exit 1
[[ $(git rev-parse --short=8 HEAD) == 2c625111 && -z $(git status --porcelain --untracked-files=no) ]] || { echo "repo not clean at 2c625111"; exit 1; }
echo "$(date '+%m-%d-%Y %H:%M:%S') check-only"
OMP_NUM_THREADS=1 "$PY" -u "$CODE/run.py" --arm gpu --check-only --workers 16 >"$OUT/logs/check_only_skampere1.log" 2>&1 || { echo "check-only failed"; exit 1; }
tail -n 1 "$OUT/logs/check_only_skampere1.log" | cut -c1-300
GPUS=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
  awk -F', ' -v m="$MIN_FREE_MB" '$2 >= m { print $3, $1 }' | sort -n | awk 'NR <= 3 { print $2 }')
[[ -n $GPUS ]] || { echo "no GPU with >= $MIN_FREE_MB MB free"; exit 1; }
echo "$(date '+%m-%d-%Y %H:%M:%S') pilot workers on GPUs $GPUS"
W=$B/pilot_worker.sh
cat >"$W" <<'WEOF'
#!/usr/bin/env bash
set -uo pipefail
G=$1 B=$2 CODE=$3 PY=$4 OUT=$5 MIN_FREE_MB=$6
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
ts() { date '+%m-%d-%Y %H:%M:%S'; }
for j in 6 7 8 3 4 5 0 1 2; do
  id=$(printf '%05d' "$j")
  [[ -f $OUT/pilot/$id/DONE ]] && continue
  mkdir "$OUT/claims_pilot/$id" 2>/dev/null || continue
  echo "skampere1 gpu$G $(ts)" >"$OUT/claims_pilot/$id/owner"
  while true; do
    free=$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    ((free >= MIN_FREE_MB)) && break
    echo "$(ts) gpu$G has ${free} MB free; waiting before pilot $id"
    sleep 120
  done
  echo "$(ts) start pilot $id on gpu$G"
  start=$(date +%s)
  "$PY" -u "$CODE/run.py" --arm gpu --pilot --pilot-job "$j" --output "$OUT/pilot" >>"$OUT/logs/pilot_$id.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\tgpu%s\t%s\t%s\n' "$(ts)" pilot "$id" "$G" "$rc" "$(($(date +%s) - start))" >>"$OUT/logs/status_skampere1.tsv"
  echo "$(ts) end pilot $id exit $rc"
done
echo "$(ts) pilot worker gpu$G finished"
WEOF
first=1
for g in $GPUS; do
  CMD="bash $W $g $B $CODE $PY $OUT $MIN_FREE_MB 2>&1 | tee -a $OUT/logs/pilot_worker_gpu$g.log"
  if ((first)); then tmux new-session -d -s tdc_exp13 -n "gpu$g" "$CMD"; tmux set-option -t tdc_exp13 remain-on-exit on >/dev/null; first=0
  else tmux new-window -t tdc_exp13 -n "gpu$g" "$CMD"; fi
done
echo "$(date '+%m-%d-%Y %H:%M:%S') SETUP_DONE"
