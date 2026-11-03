#!/usr/bin/env bash
# Experiment 24 autopilot on SNAP skampere2 (tmux session tdc_exp24). Other people's work and Laolu's other runs take
# precedence: a job starts only on a GPU with >= MIN_FREE_MB free and <= MAX_UTIL % utilization (worst of 3 samples,
# 20 s apart), re-checked before every job, at most MAX_JOBS of ours at once; otherwise it waits. skampere1 is never
# used. Stages (each resumable; state files in $STATE):
#   0 unit tests (CPU) and prep (run24.py --check-only, CPU)
#   1 GPU smoke: 2 shards' jobs to 2^22 (one S alpha = 1, one M disjoint alpha = 0.01), each verified
#   2 pilot (6 jobs), pilot rule, "Experiment 24 pilot disclosure" commit in this clone (not pushed from here)
#   3 the 918 shards, M first; each verified on its GPU right after training
#   4 merge, analyze24, task2vec24, verify24 (CPU); ALL_DONE
# A failure writes $STATE/HALT and stops at the end of the stage. No automatic reruns. touch $STATE/STOP to stop
# launching new jobs (running ones finish).
set -uo pipefail
U=laoluog1
B=/dfs/scratch1/$U/tdc_exp24
REPO=$B/the-difficulty-confound
EXP=experiments/24_similarity_alignment_complexity
CODE=$REPO/$EXP/expt_v1
PY=/lfs/skampere2/0/$U/venvs/tdc_exp13/bin/python
OUT=$B/runs
STATE=$B/state
LOG=$OUT/logs/autopilot24.log
MIN_FREE_MB=${MIN_FREE_MB:-24000}
MAX_UTIL=${MAX_UTIL:-30}
MAX_JOBS=${MAX_JOBS:-4}
NSHARD=918
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv TORCH_HOME=$B/cache/torch \
  MPLCONFIGDIR=$B/cache/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$STATE" "$OUT/logs" "$OUT/shards" "$OUT/claims" "$OUT/pilot" "$OUT/smoke" "$B/cache"
say() { echo "$(date '+%m-%d-%Y %H:%M:%S') $*" | tee -a "$LOG"; }
halt() { say "HALT: $*"; echo "$(date '+%m-%d-%Y %H:%M:%S') $*" >"$STATE/HALT"; exit 1; }
[[ $(hostname -s) == skampere2 ]] || halt "run this on skampere2 only"
[[ ! -f $STATE/HALT ]] || { say "HALT present: $(cat "$STATE/HALT")"; exit 1; }
cd "$REPO" || halt "no repo"

# ---- GPU queue -----------------------------------------------------------------------------------------------
declare -A PID_GPU=()
reap() {
  local p
  for p in "${!PID_GPU[@]}"; do
    kill -0 "$p" 2>/dev/null || unset "PID_GPU[$p]"
  done
}
gpu_free() {  # prints a qualifying GPU index not used by our running jobs, or nothing
  local s1 s2 s3 g used
  s1=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits); sleep 20
  s2=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits); sleep 20
  s3=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  used=" ${PID_GPU[*]-} "
  printf '%s\n%s\n%s\n' "$s1" "$s2" "$s3" | awk -F', ' '{ g=$1; if (!(g in mf) || $2 < mf[g]) mf[g]=$2; if ($3 > mu[g]) mu[g]=$3 }
    END { for (g in mf) print mu[g], mf[g], g }' | sort -n | while read -r u f g; do
    if ((f >= MIN_FREE_MB && u <= MAX_UTIL)) && [[ $used != *" $g "* ]]; then echo "$g"; break; fi
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
    if ((${#PID_GPU[@]} < MAX_JOBS)); then
      g=$(gpu_free)
      [[ -n $g ]] && { echo "$g"; return; }
    fi
    sleep 60
  done
}
wait_all() { while true; do reap; ((${#PID_GPU[@]} == 0)) && return; sleep 30; done; }

job_script() {  # job_script <kind> <id> <outdir>: train (or pilot/smoke) then verify, then a status line
  local kind=$1 id=$2 o=$3 pad
  pad=$(printf '%05d' "$id")
  cat <<EOF
cd "$REPO"; s=\$(date +%s)
case $kind in
  shard) "$PY" -u "$CODE/run24.py" --shard $id --output "$o" >>"$OUT/logs/shard_$pad.log" 2>&1 ;;
  smoke) "$PY" -u "$CODE/run24.py" --smoke $id --output "$OUT" >>"$OUT/logs/smoke_$pad.log" 2>&1 ;;
  pilot) "$PY" -u "$CODE/run24.py" --pilot-job $id --output "$o" >>"$OUT/logs/pilot_$pad.log" 2>&1 ;;
esac
rc=\$?; vrc=-
if [[ \$rc == 0 && $kind != pilot ]]; then
  vo="$o"; [[ $kind == smoke ]] && vo="$OUT/smoke"
  "$PY" -u "$CODE/verify24.py" --shard $id --output "\$vo" >>"$OUT/logs/${kind}_verify_$pad.log" 2>&1; vrc=\$?
fi
printf '%s\t%s\t%s\tgpu%s\t%s\t%s\t%s\n' "\$(date '+%m-%d-%Y %H:%M:%S')" $kind $pad "\$CUDA_VISIBLE_DEVICES" \$rc \$vrc \$((\$(date +%s) - s)) >>"$OUT/logs/status.tsv"
EOF
}

# ---- 0. tests and prep ----------------------------------------------------------------------------------------
if [[ ! -f $STATE/PREP_DONE ]]; then
  say "unit tests (CPU)"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q . >"$OUT/logs/pytest24.log" 2>&1) || halt "Exp 24 unit tests failed"
  (cd "$REPO/experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1" && CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q . \
    >"$OUT/logs/pytest13.log" 2>&1) || halt "Exp 13 unit tests failed"
  say "tests pass: $(tail -n 1 "$OUT/logs/pytest24.log"); $(tail -n 1 "$OUT/logs/pytest13.log")"
  say "prep (run24.py --check-only)"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PY" -u "$CODE/run24.py" --check-only --workers 8 >"$OUT/logs/check24.log" 2>&1 ||
    halt "check-only failed"
  cp "$CODE/results_check.json" "$OUT/logs/results_check_skampere2.json"  # the receipt of record for this prep
  git checkout -- "$EXP/expt_v1/results_check.json" 2>/dev/null  # keep the tracked (laptop) receipt unmodified
  [[ -z $(git status --porcelain --untracked-files=no) ]] || halt "tracked files modified after check-only"
  touch "$STATE/PREP_DONE"
  say "prep done"
fi

# ---- 1. GPU smoke -----------------------------------------------------------------------------------------------
if [[ ! -f $STATE/SMOKE_DONE ]]; then
  for id in 0 906; do
    [[ -f $OUT/smoke/shards/$(printf '%05d' $id)/VERIFIED ]] && continue
    g=$(wait_slot); [[ $g == STOP ]] && halt "STOP requested"
    say "smoke $id on gpu$g"
    launch "$g" "$OUT/logs/worker.log" bash -c "$(job_script smoke $id "$OUT")"
  done
  wait_all
  for id in 0 906; do
    [[ -f $OUT/smoke/shards/$(printf '%05d' $id)/VERIFIED ]] || halt "smoke $id failed (logs/smoke_*.log)"
  done
  touch "$STATE/SMOKE_DONE"
  say "GPU smoke: 2 of 2 trained and VERIFIED"
fi

# ---- 2. pilot and the disclosure commit -----------------------------------------------------------------------
if [[ ! -f $STATE/DISCLOSURE_SHA ]]; then
  for j in 3 4 5 0 1 2; do
    [[ -f $OUT/pilot/$(printf '%05d' $j)/DONE ]] && continue
    g=$(wait_slot); [[ $g == STOP ]] && halt "STOP requested"
    say "pilot $j on gpu$g"
    launch "$g" "$OUT/logs/worker.log" bash -c "$(job_script pilot $j "$OUT")"
  done
  wait_all
  "$PY" "$CODE/run24.py" --pilot-select "$OUT" >>"$LOG" 2>&1 || halt "pilot selection failed"
  "$PY" - "$OUT/pilot/pilot_selection.json" "$REPO/$EXP" >>"$LOG" 2>&1 <<'PYEOF' || halt "disclosure edit failed"
import json, sys, time
from pathlib import Path
sel = json.loads(Path(sys.argv[1]).read_text()); exp = Path(sys.argv[2])
ch = sel['lr_selected']; res = sel['results']
cfg = exp / 'expt_v1' / 'config.json'; s = cfg.read_text()
assert s.count('"lr_selected": null') == 1
cfg.write_text(s.replace('"lr_selected": null', '"lr_selected": ' + json.dumps(ch)))
f = lambda z: ', '.join(f"{r['lr_multiplier']}x {r['excess_bits']:.4f}" for r in res if r['size'] == z)
par = (f"\n**Pilot disclosure ({time.strftime('%m-%d-%Y')}, SNAP skampere2, {', '.join(sorted({r['gpu'] for r in res}))}).** "
       f"Seed `24919`, the n = 32 pilot concept with its family pool, alpha = 0.2, T to 2^26, 6 runs. Excess at 2^26 "
       f"(bits/token, all positions): S {f('S')}; M {f('M')}. Rule of Section 2.5: **S {ch['S']}x, M {ch['M']}x** of the "
       f"13-GPU table rate (`lr_selected`). Seed `24919` is never used again. This commit changes only `lr_selected` and "
       f"adds this paragraph; every shard runs at it.\n")
p = exp / 'PROTOCOL.md'; t = p.read_text()
anchor = '(Recorded here, with dates, by the pilot disclosure commit and any later deviation.)\n'
assert t.count(anchor) == 1
p.write_text(t.replace(anchor, anchor + par))
print('disclosure edits written', ch)
PYEOF
  git add "$EXP/expt_v1/config.json" "$EXP/PROTOCOL.md" || halt "git add failed"
  [[ $(git diff --cached --name-only | wc -l) == 2 ]] || halt "disclosure commit would touch other files"
  git -c user.name="Laolu Oguneye" -c user.email=laoluog@outlook.com commit -q -m "Experiment 24 pilot disclosure" \
    -m "Pilot (seed 24919, n = 32 pilot concept, family pool, alpha 0.2, T to 2^26) on skampere2; frozen rule of PROTOCOL Section 2.5. Changes only config.json lr_selected and adds the pilot disclosure paragraph to PROTOCOL Section 8. Every shard runs at this commit." ||
    halt "disclosure commit failed"
  git rev-parse HEAD >"$STATE/DISCLOSURE_SHA"
  say "disclosure commit $(cat "$STATE/DISCLOSURE_SHA"): $(grep -o '"lr_selected": {[^}]*}' "$CODE/config.json")"
fi
[[ $(git rev-parse HEAD) == $(cat "$STATE/DISCLOSURE_SHA") ]] || halt "repo HEAD is not the disclosure commit"
[[ -z $(git status --porcelain --untracked-files=no) ]] || halt "tracked files modified"

# ---- 3. shards (M first) -----------------------------------------------------------------------------------------
if [[ ! -f $STATE/SHARDS_DONE ]]; then
  say "shards: queue of $NSHARD (M first)"
  for id in $(seq 459 917) $(seq 0 458); do
    pad=$(printf '%05d' "$id")
    [[ -f $OUT/shards/$pad/VERIFIED ]] && continue
    [[ -f $OUT/shards/$pad/DONE ]] && continue
    mkdir "$OUT/claims/$pad" 2>/dev/null || continue
    g=$(wait_slot)
    if [[ $g == STOP ]]; then rmdir "$OUT/claims/$pad"; say "STOP: no new shards launched"; break; fi
    echo "skampere2 gpu$g $(date '+%m-%d-%Y %H:%M:%S')" >"$OUT/claims/$pad/owner"
    launch "$g" "$OUT/logs/worker.log" bash -c "$(job_script shard "$id" "$OUT")"
    n=$(ls "$OUT"/shards/*/VERIFIED 2>/dev/null | wc -l)
    say "shard $id on gpu$g ($n VERIFIED so far)"
  done
  wait_all
  nv=$(ls "$OUT"/shards/*/VERIFIED 2>/dev/null | wc -l)
  say "shards finished: $nv of $NSHARD VERIFIED"
  ((nv == NSHARD)) || halt "$nv of $NSHARD shards VERIFIED: failed shards need a manual rerun (claims kept; logs/status.tsv)"
  touch "$STATE/SHARDS_DONE"
fi

# ---- 4. merge, analysis, Task2Vec, merged verify (CPU) --------------------------------------------------------------
RES=$CODE/results
if [[ ! -f $RES/manifest.json ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" -u "$CODE/run24.py" --merge "$OUT" >>"$LOG" 2>&1 || halt "merge failed"
fi
cd "$CODE" || halt "no code dir"
if [[ ! -f $RES/task2vec.json ]]; then
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 "$PY" -u task2vec24.py --results "$RES" --device cpu >"$OUT/logs/task2vec.log" 2>&1 ||
    say "task2vec failed (reported only; continuing): see logs/task2vec.log"
fi
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PY" -u analyze24.py --results "$RES" --workers 8 >"$OUT/logs/analyze24.log" 2>&1 ||
  halt "analysis failed"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PY" -u verify24.py --results "$RES" --workers 8 >"$OUT/logs/verify24_merged.log" 2>&1 ||
  halt "merged verify failed (logs/verify24_merged.log)"
touch "$STATE/ALL_DONE"
say "ALL_DONE: $(tr -d '\n' <"$RES/gates.json" | grep -o '"verdicts": {[^}]*}')"
