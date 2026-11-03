#!/bin/bash
# Experiment 23 full grid on FarmShare (PROTOCOL Sections 3 and 7). Run once, after prepare.sh and the smoke, from a
# shell where sbatch works (the JupyterLab terminal):
#   bash ~/e23/code/experiments/23_exp13_full_training_8seeds/farmshare/submit_full.sh --dry-run   # print, submit nothing
#   bash ~/e23/code/experiments/23_exp13_full_training_8seeds/farmshare/submit_full.sh             # submit the chain
# Resume after a failure (a decide job exits 1 when runs of its stage are missing; finished runs are never redone):
#   bash .../submit_full.sh --from-stage K [--exp e13|e17b]
#
# Chain per experiment, stages k = FROM..3 (T = 2^20, 2^22, 2^24, 2^26):
#   array  e23_<exp>_s<k>  (64 tasks for e13 at %24, 40 for e17b at %16; 4 runs per task, -c 4)  afterok: decide k-1
#   decide e23_<exp>_d<k>  afterany: array k, afterok: decide k-1      (writes STOP when the grid has converged;
#                                                                       later arrays then exit at once)
#   then   e23_analyze     afterok: both experiments' last decide      (aggregate, analyses, verify, RESULTS.md)
# About 425 jobs, under the normal QOS limit of 1024 submitted; at most 24 + 16 = 40 array tasks run at once (<= 128).
# Mail: arrays and analyze --mail-type=END,FAIL (one mail per array, never per task); decide jobs FAIL only.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
cd "$E23_CODE"
FROM=0; DRY=0; EXPS="e13 e17b"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --from-stage) FROM=$2; shift ;;
    --exp) EXPS=$2; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
  shift
done
cfg() { "$E23_PY" -c "import json; c = json.load(open('config.json')); print($1)"; }
NSTAGE=$(cfg "len(c['stages_T'])")
read -r -a TIMES <<< "$(cfg "' '.join(c['farmshare']['time_per_stage'])")"
PART=$(cfg "c['farmshare']['partition']")
LIMIT=$(cfg "c['farmshare']['qos_submit_limit']")

to_min() {  # Slurm time [D-]HH:MM:SS -> minutes; "infinite"/"UNLIMITED" -> a large number
  local t=$1 d=0
  case "$t" in infinite|UNLIMITED|"") echo 999999; return ;; esac
  if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi
  IFS=: read -r -a p <<< "$t"
  if [ ${#p[@]} -eq 3 ]; then echo $(( d*1440 + 10#${p[0]}*60 + 10#${p[1]} )); else echo $(( d*1440 + 10#${p[0]} )); fi
}
PLIM=$(sinfo -p "$PART" -h -o %l | head -1)
for k in $(seq "$FROM" $((NSTAGE-1))); do
  if [ "$(to_min "${TIMES[$k]}")" -gt "$(to_min "$PLIM")" ]; then
    echo "NOTE: stage $k asks ${TIMES[$k]} but partition $PART allows $PLIM: capped to $PLIM. A task that times out"
    echo "      resumes from its last checkpoint when you rerun with --from-stage $k."
    TIMES[$k]=$PLIM
  fi
done
echo "partition $PART time limit $PLIM; stage limits ${TIMES[*]}"

declare -A NT THR
NEED=1
for exp in $EXPS; do
  NT[$exp]=$("$E23_PY" run23.py count --exp $exp --prep "$E23_OUT" 2>/dev/null | awk '{print $2}' || true)
  [ -n "${NT[$exp]}" ] || NT[$exp]=$("$E23_PY" run23.py count --exp $exp | awk '{print $2}')
  THR[$exp]=$(cfg "c['farmshare']['throttle']['$exp']")
  NEED=$(( NEED + (NSTAGE - FROM) * (NT[$exp] + 1) ))
done
CUR=$(squeue -r -h -u "$USER" -q normal 2>/dev/null | wc -l)
echo "normal QOS: $CUR jobs of yours now, this submit adds up to $NEED, limit $LIMIT"
if [ $(( CUR + NEED )) -gt "$LIMIT" ]; then echo "would exceed the submit limit; nothing submitted"; exit 4; fi

N=0
sb() {  # sbatch --parsable, or print in --dry-run; $1 = the job name, also the dry-run id
  local name=$1; shift
  if [ $DRY -eq 1 ]; then echo "sbatch -J $name $*" >&2; echo "<$name>"; else sbatch --parsable -J "$name" "$@" | cut -d';' -f1; fi
}
STAMP=$(date +%Y%m%d_%H%M%S)
IDS="$E23_HOME/job_ids_$STAMP.txt"; [ $DRY -eq 1 ] && IDS=/dev/null
declare -A LAST
for exp in $EXPS; do
  if [ -f "$E23_OUT/$exp/STOP" ]; then echo "$exp already stopped: $(cat "$E23_OUT/$exp/STOP")"; continue; fi
  prev=""
  for k in $(seq "$FROM" $((NSTAGE-1))); do
    dep=(); [ -n "$prev" ] && dep=(--dependency=afterok:$prev --kill-on-invalid-dep=yes)
    arr=$(sb e23_${exp}_s$k -p "$PART" --array=0-$(( NT[$exp] - 1 ))%${THR[$exp]} -t "${TIMES[$k]}" \
             --mail-type=END,FAIL --mail-user="$E23_MAIL" -o "$E23_LOGS/%x_%A_%a.out" "${dep[@]}" \
             farmshare/stage.sbatch $exp $k)
    ddep="afterany:$arr"; [ -n "$prev" ] && ddep="$ddep,afterok:$prev"
    dec=$(sb e23_${exp}_d$k -p "$PART" --dependency=$ddep --kill-on-invalid-dep=yes --mail-type=FAIL \
             --mail-user="$E23_MAIL" -o "$E23_LOGS/%x_%j.out" farmshare/decide.sbatch $exp $k)
    N=$((N+2))
    echo "$exp stage $k: array $arr, decide $dec" | tee -a "$IDS"
    prev=$dec
  done
  LAST[$exp]=$prev
done
# the analysis waits for both experiments: this submit's last decide jobs, or one still queued from an earlier submit
deps=""
for exp in e13 e17b; do
  j=${LAST[$exp]:-}
  if [ -z "$j" ] && [ $DRY -eq 0 ]; then j=$(squeue -h -u "$USER" -n e23_${exp}_d$((NSTAGE-1)) -o %i | head -1); fi
  if [ -z "$j" ] && [ ! -f "$E23_OUT/$exp/STOP" ] && [ $DRY -eq 0 ]; then
    echo "WARNING: $exp has no STOP and no queued decide job; submit it with --exp $exp before the analysis can run"
  fi
  [ -n "$j" ] && deps="$deps:$j"
done
adep=(); [ -n "$deps" ] && adep=(--dependency=afterok$deps --kill-on-invalid-dep=yes)
N=$((N+1))
an=$(sb e23_analyze -p "$PART" -t 08:00:00 "${adep[@]}" --mail-type=END,FAIL --mail-user="$E23_MAIL" \
        -o "$E23_LOGS/%x_%j.out" farmshare/analyze.sbatch)
echo "analyze: $an" | tee -a "$IDS"
echo "submitted $N sbatch calls (dry run: $DRY); ids in $IDS"
