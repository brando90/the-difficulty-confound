#!/usr/bin/env bash
# GPU plans g1-g3 on ONE SNAP GPU node (skampere1/2, or mercury1/2 for the 0.5B tiers only): one worker per free
# GPU in the tmux session tdc_gpu, each taking registry indices round-robin. Run it ON the node after logging in once
# (never retry a failed login in a loop: whale bans IPs). It never opens SSH itself.
#
#   cd /dfs/scratch0/$USER/tdc_gpu/the-difficulty-confound
#   MODE=prep TDC_COMMIT=<sha> bash experiments/gpu/launch/snap.sh <host>            # once: data + models (CPU)
#   TDC_COMMIT=<sha> TIER=main PLANS=g1,g2,g3 bash experiments/gpu/launch/snap.sh <host> [first] [last]
#
# Optional: SEEDS=0, SECONDARY=1, GPUS=0,1 (restrict), MAX_GPUS, MIN_FREE_MB (default 30000), BASE, TORCH_INDEX.
set -euo pipefail
die() { echo "snap.sh: ERROR: $*" >&2; exit 1; }
[[ $# -ge 1 ]] || die "usage: [MODE=prep] TDC_COMMIT=<sha> TIER=<tier> PLANS=<plans> bash $0 <host> [first] [last]"
HOST=$1
[[ $(hostname -s) == "$HOST" ]] || die "this is $(hostname -s), not $HOST. Log in to $HOST once and rerun there."
BASE=${BASE:-/dfs/scratch0/$USER/tdc_gpu}
REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
source "$REPO/experiments/gpu/launch/common_env.sh"
check_commit
if [[ ${MODE:-run} == prep ]]; then
  "$PY" -u "$GPU_DIR/common/prep_data.py" --data-root "$DATA"
  "$PY" -c 'from huggingface_hub import snapshot_download as s; [print(s(m)) for m in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]]'
  exit 0
fi
: "${TIER:?set TIER=pilot|main|scale}"
PLANS=${PLANS:-g1,g2,g3}
[[ -f $DATA/v1/manifest.json ]] || die "no data at $DATA; run MODE=prep first"
EXTRA=""
[[ ${SECONDARY:-0} == 1 ]] && EXTRA="--secondary"
N=$("$PY" "$GPU_DIR/common/runs.py" count --tier "$TIER" --plans "$PLANS" --seeds "${SEEDS:-all}" $EXTRA)
FIRST=${2:-0}
LAST=${3:-$((N - 1))}
((LAST < N)) || die "last ($LAST) must be < $N"
command -v tmux >/dev/null || die "tmux not found"
tmux has-session -t tdc_gpu 2>/dev/null && die "tmux session tdc_gpu already exists; attach with 'tmux attach -t tdc_gpu'"
FREE=()
while IFS=',' read -r idx used free util; do
  idx=${idx// /}; used=${used// /}; free=${free// /}; util=${util// /}
  [[ -n ${GPUS:-} && ",$GPUS," != *",$idx,"* ]] && continue
  if ((used <= 2000 && util <= 10 && free >= ${MIN_FREE_MB:-30000})); then FREE+=("$idx"); fi
done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
((${#FREE[@]} > 0)) || die "no free GPU on $HOST"
FREE=("${FREE[@]:0:${MAX_GPUS:-8}}")
echo "snap.sh: GPUs ${FREE[*]} ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)); indices $FIRST..$LAST of $N"
mkdir -p "$OUT/logs"
tmux new-session -d -s tdc_gpu -n main "echo tdc_gpu workers; sleep infinity"
NG=${#FREE[@]}
for ((k = 0; k < NG; k++)); do
  g=${FREE[$k]}
  IDS=""
  for ((i = FIRST + k; i <= LAST; i += NG)); do IDS+="$i "; done
  [[ -n $IDS ]] || continue
  W=$OUT/logs/worker_${HOST}_gpu${g}.sh
  cat >"$W" <<WEOF
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $REPO
for i in $IDS; do
  start=\$(date +%s)
  $PY -u $GPU_DIR/common/train.py --tier $TIER --plans $PLANS --seeds ${SEEDS:-all} $EXTRA --index \$i --data-root $DATA --out-root $OUT >>$OUT/logs/${TIER}_${PLANS//,/-}_\$(printf '%04d' \$i).log 2>&1
  rc=\$?
  printf '%s\t%s\t%s\t%s\tsnap:%s:gpu%s\t%s\t%s\n' "\$(date '+%m-%d-%Y %H:%M:%S')" $TIER $PLANS \$i $HOST $g \$rc \$((\$(date +%s) - start)) >>$OUT/logs/status.tsv
done
echo "worker gpu$g finished"
WEOF
  tmux new-window -t tdc_gpu -n "gpu$g" "bash $W; sleep infinity"
done
echo "snap.sh: started $NG workers. Watch: tmux attach -t tdc_gpu; progress: $PY $GPU_DIR/common/runs.py status --tier $TIER --plans $PLANS --out-root $OUT"
