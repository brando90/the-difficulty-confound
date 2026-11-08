#!/usr/bin/env bash
# Experiment 25 autopilot on SNAP skampere2 (tmux session tdc_exp25). skampere1 is never used. Other people's work
# takes precedence: a job starts only on a GPU with >= MIN_FREE_MB free (worst of 3 samples, 20 s apart), re-checked
# before every job, at most JOBS_PER_GPU of ours per GPU and MAX_JOBS of ours at once; otherwise it waits (PROTOCOL
# Section 7). Stages (each resumable; state files in $STATE):
#   0 unit tests (CPU) and prep (run25.py --check-only, CPU)
#   1 GPU smoke: 2 shards' jobs to 2^22 (S alpha = 1 at 0.5x; M masked alpha = 0.02 at 8x), each verified
#   2 the 2,880 shards, M first; each verified on its GPU right after training
#   3 merge (CPU), then WAIT for $STATE/PREDICTIONS_SHA (the commit with Laolu's predictions; PROTOCOL Section 5)
#   4 analyze25, verify25 (CPU); ALL_DONE
# A failure writes $STATE/HALT and stops at the end of the stage. No automatic reruns. touch $STATE/STOP to stop
# launching new jobs (running ones finish). $STATE/SKIP_GPUS (space-separated indices) keeps the queue off those GPUs;
# $STATE/MAX_JOBS, JOBS_PER_GPU, MIN_FREE_MB and MAX_UTIL (integers) override the defaults. All are re-read every
# cycle, so they change without a restart (a restart kills running jobs and strands their claims).
set -uo pipefail
U=laoluog1
B=/dfs/scratch1/$U/tdc_exp25
REPO=$B/the-difficulty-confound
EXP=experiments/25_alpha_elasticity_diagnosis
CODE=$REPO/$EXP/expt_v1
PY=/lfs/skampere2/0/$U/venvs/tdc_exp13/bin/python
OUT=$B/runs
STATE=$B/state
LOG=$OUT/logs/autopilot25.log
MIN_FREE_MB=${MIN_FREE_MB:-24000}
MAX_UTIL=${MAX_UTIL:-100}
MAX_JOBS=${MAX_JOBS:-12}
JOBS_PER_GPU_DEFAULT=3
NSHARD=2880
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv TORCH_HOME=$B/cache/torch \
  MPLCONFIGDIR=$B/cache/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$STATE" "$OUT/logs" "$OUT/shards" "$OUT/claims" "$OUT/smoke" "$B/cache"
say() { echo "$(date '+%m-%d-%Y %H:%M:%S') $*" | tee -a "$LOG"; }
halt() { say "HALT: $*"; echo "$(date '+%m-%d-%Y %H:%M:%S') $*" >"$STATE/HALT"; exit 1; }
[[ $(hostname -s) == skampere2 ]] || halt "run this on skampere2 only"
[[ ! -f $STATE/HALT ]] || { say "HALT present: $(cat "$STATE/HALT")"; exit 1; }
[[ -f $STATE/FREEZE_SHA ]] || halt "no state/FREEZE_SHA (setup25.sh writes it)"
cd "$REPO" || halt "no repo"
[[ $(git rev-parse HEAD) == $(cat "$STATE/FREEZE_SHA") ]] || halt "repo HEAD is not the freeze commit"
[[ -z $(git status --porcelain --untracked-files=no) ]] || halt "tracked files modified"

# ---- GPU queue -----------------------------------------------------------------------------------------------
declare -A PID_GPU=()
reap() {
  local p
  for p in "${!PID_GPU[@]}"; do
    kill -0 "$p" 2>/dev/null || unset "PID_GPU[$p]"
  done
}
knob() {  # knob <name> <default>: an integer from $STATE/<name> if set there (re-read every cycle), else the default
  local v
  v=$(tr -d ' \n' 2>/dev/null <"$STATE/$1")
  [[ $v =~ ^[0-9]+$ ]] && echo "$v" || echo "$2"
}
gpu_free() {  # prints a qualifying GPU index, or nothing. A GPU without our jobs must pass the memory and utilization
  # rules; one already running ours (its utilization is then partly ours) needs the memory rule and < JOBS_PER_GPU of ours
  local s1 s2 s3 skip per ours x minf maxu
  s1=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits); sleep 20
  s2=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits); sleep 20
  s3=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  skip=" $(tr '\n' ' ' 2>/dev/null <"$STATE/SKIP_GPUS") "  # SKIP_GPUS: indices held for other work
  per=$(knob JOBS_PER_GPU "$JOBS_PER_GPU_DEFAULT"); minf=$(knob MIN_FREE_MB "$MIN_FREE_MB"); maxu=$(knob MAX_UTIL "$MAX_UTIL")
  printf '%s\n%s\n%s\n' "$s1" "$s2" "$s3" | awk -F', ' '{ g=$1; if (!(g in mf) || $2 < mf[g]) mf[g]=$2; if (!(g in mu) || $3 > mu[g]) mu[g]=$3 }
    END { for (g in mf) print mu[g], mf[g], g }' | sort -n | while read -r u f g; do
    [[ $skip == *" $g "* ]] && continue
    ours=0
    for x in ${PID_GPU[@]+"${PID_GPU[@]}"}; do [[ $x == "$g" ]] && ours=$((ours + 1)); done
    if ((f >= minf && ours < per && (ours > 0 || u <= maxu))); then echo "$g"; break; fi
  done
}
launch() {  # launch <gpu> <log> <command...>: runs in the background on that GPU
  local g=$1 lg=$2
  shift 2
  (export CUDA_VISIBLE_DEVICES=$g; "$@" >>"$lg" 2>&1) &
  PID_GPU[$!]=$g
}
wait_slot() {  # blocks until we have < MAX_JOBS jobs and a GPU qualifies; prints it
  local g
  while true; do
    reap
    [[ -f $STATE/STOP ]] && { echo STOP; return; }
    if ((${#PID_GPU[@]} < $(knob MAX_JOBS "$MAX_JOBS"))); then
      g=$(gpu_free)
      [[ -n $g ]] && { echo "$g"; return; }
    fi
    sleep 60
  done
}
wait_all() { while true; do reap; ((${#PID_GPU[@]} == 0)) && return; sleep 30; done; }

job_script() {  # job_script <kind> <id>: train (shard or smoke), then verify on the same GPU, then a status line
  local kind=$1 id=$2 pad
  pad=$(printf '%05d' "$id")
  cat <<EOF
cd "$REPO"; s=\$(date +%s)
case $kind in
  shard) "$PY" -u "$CODE/run25.py" --shard $id --output "$OUT" >>"$OUT/logs/shard_$pad.log" 2>&1 ;;
  smoke) "$PY" -u "$CODE/run25.py" --smoke $id --output "$OUT" >>"$OUT/logs/smoke_$pad.log" 2>&1 ;;
esac
rc=\$?; vrc=-
if [[ \$rc == 0 ]]; then
  vf=; [[ $kind == smoke ]] && vf=--smoke
  "$PY" -u "$CODE/verify25.py" --shard $id \$vf --output "$OUT" >>"$OUT/logs/${kind}_verify_$pad.log" 2>&1; vrc=\$?
fi
printf '%s\t%s\t%s\tgpu%s\t%s\t%s\t%s\n' "\$(date '+%m-%d-%Y %H:%M:%S')" $kind $pad "\$CUDA_VISIBLE_DEVICES" \$rc \$vrc \$((\$(date +%s) - s)) >>"$OUT/logs/status.tsv"
EOF
}

# ---- 0. tests and prep ----------------------------------------------------------------------------------------
if [[ ! -f $STATE/PREP_DONE ]]; then
  say "unit tests (CPU)"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q . >"$OUT/logs/pytest25.log" 2>&1) || halt "Exp 25 unit tests failed"
  (cd "$REPO/experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1" && CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q . \
    >"$OUT/logs/pytest13.log" 2>&1) || halt "Exp 13 unit tests failed"
  say "tests pass: $(tail -n 1 "$OUT/logs/pytest25.log"); $(tail -n 1 "$OUT/logs/pytest13.log")"
  say "prep (run25.py --check-only)"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PY" -u "$CODE/run25.py" --check-only --workers 8 >"$OUT/logs/check25.log" 2>&1 ||
    halt "check-only failed"
  cp "$CODE/results_check.json" "$OUT/logs/results_check_skampere2.json"  # the receipt of record for this prep
  git checkout -- "$EXP/expt_v1/results_check.json" 2>/dev/null  # keep the tracked (laptop) receipt unmodified
  [[ -z $(git status --porcelain --untracked-files=no) ]] || halt "tracked files modified after check-only"
  touch "$STATE/PREP_DONE"
  say "prep done"
fi

# ---- 1. GPU smoke -----------------------------------------------------------------------------------------------
SMOKE=$(cd "$CODE" && CUDA_VISIBLE_DEVICES= "$PY" -c "import run25 as R, concepts25 as C; print(*R.smoke_shards(C.load_config()))") ||
  halt "cannot list the smoke shards"
if [[ ! -f $STATE/SMOKE_DONE ]]; then
  for id in $SMOKE; do
    [[ -f $OUT/smoke/shards/$(printf '%05d' "$id")/VERIFIED ]] && continue
    g=$(wait_slot); [[ $g == STOP ]] && halt "STOP requested"
    say "smoke $id on gpu$g"
    launch "$g" "$OUT/logs/worker.log" bash -c "$(job_script smoke "$id")"
  done
  wait_all
  for id in $SMOKE; do
    [[ -f $OUT/smoke/shards/$(printf '%05d' "$id")/VERIFIED ]] || halt "smoke $id failed (logs/smoke_*.log)"
  done
  touch "$STATE/SMOKE_DONE"
  say "GPU smoke: 2 of 2 trained and VERIFIED ($SMOKE)"
fi

# ---- 2. shards (M first) -----------------------------------------------------------------------------------------
if [[ ! -f $STATE/SHARDS_DONE ]]; then
  say "shards: queue of $NSHARD (M first)"
  for id in $(seq 1440 2879) $(seq 0 1439); do
    pad=$(printf '%05d' "$id")
    [[ -f $OUT/shards/$pad/VERIFIED ]] && continue
    [[ -f $OUT/shards/$pad/DONE ]] && continue
    mkdir "$OUT/claims/$pad" 2>/dev/null || continue
    g=$(wait_slot)
    if [[ $g == STOP ]]; then rmdir "$OUT/claims/$pad"; say "STOP: no new shards launched"; break; fi
    echo "skampere2 gpu$g $(date '+%m-%d-%Y %H:%M:%S')" >"$OUT/claims/$pad/owner"
    launch "$g" "$OUT/logs/worker.log" bash -c "$(job_script shard "$id")"
    n=$(ls "$OUT"/shards/*/VERIFIED 2>/dev/null | wc -l)
    say "shard $id on gpu$g ($n VERIFIED so far)"
  done
  wait_all
  nv=$(ls "$OUT"/shards/*/VERIFIED 2>/dev/null | wc -l)
  say "shards finished: $nv of $NSHARD VERIFIED"
  ((nv == NSHARD)) || halt "$nv of $NSHARD shards VERIFIED: failed shards need a manual rerun (claims kept; logs/status.tsv)"
  touch "$STATE/SHARDS_DONE"
fi

# ---- 3. merge, then wait for the predictions commit ---------------------------------------------------------------
RES=$CODE/results
if [[ ! -f $RES/manifest.json ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -u "$CODE/run25.py" --merge "$OUT" >>"$LOG" 2>&1 || halt "merge failed"
  say "merged; waiting for state/PREDICTIONS_SHA before the analysis (PROTOCOL Section 5)"
fi
until [[ -s $STATE/PREDICTIONS_SHA ]]; do sleep 300; done
say "predictions commit $(cat "$STATE/PREDICTIONS_SHA"); analysis starts"

# ---- 4. analysis and merged verify (CPU) -----------------------------------------------------------------------------
cd "$CODE" || halt "no code dir"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PY" -u analyze25.py --results "$RES" --workers 8 >"$OUT/logs/analyze25.log" 2>&1 ||
  halt "analysis failed"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PY" -u verify25.py --results "$RES" --workers 8 >"$OUT/logs/verify25_merged.log" 2>&1 ||
  halt "merged verify failed (logs/verify25_merged.log)"
touch "$STATE/ALL_DONE"
say "ALL_DONE: $(tr -d '\n' <"$RES/gates.json" | grep -o '"verdicts": {[^}]*}')"
