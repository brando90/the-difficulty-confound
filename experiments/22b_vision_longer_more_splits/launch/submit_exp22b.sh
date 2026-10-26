#!/bin/bash
# Submit Experiment 22b (vision rerun: 300 epochs, 10 splits) on FarmShare, once. From the JupyterLab terminal (a
# compute node with sbatch and internet), with the experiment checked out at the frozen commit:
#   mkdir -p ~/exp22b && cd ~/exp22b && git clone https://github.com/brando90/the-difficulty-confound && cd the-difficulty-confound \
#     && git checkout <FREEZE commit> && bash experiments/22b_vision_longer_more_splits/launch/submit_exp22b.sh 2>&1 | tee ~/exp22b/submit_exp22b.log
# 1) prep: `sbatch --wait` on -p normal. It reuses Experiment 22's venv ~/venvs/exp22 (.ready) and data ~/exp22/{raw,data}
#    when they exist (rebuilds them if not), re-checks every hash and the split tests; nothing else is submitted
#    unless it prints EXP22B PREP DONE.
# 2) one GPU array of all 81 runs (feat + 40 vis_real + 40 vis_shuf), packed into as many tasks as the gpu QOS has free
#    submit slots allow (at most 25, at least 6; the post job needs one), on -p gpu --qos=gpu (L40S, bf16), %4.
# 3) a post job (GPU) after the array ends: verify.py --stage remote, then ~/exp22b/exp22b_results.tgz.
# Slurm mail END,FAIL to both of Brando's addresses on every job (one mail per array, never per task).
set -euo pipefail
BASE=$HOME/exp22b
MAIL=(--mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com)
GPU=(-p gpu --qos=gpu --gres=gpu:1)
export EXP22_VENV=$HOME/venvs/exp22
export EXP22_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$BASE"
[ -e "$BASE/jobs_exp22b.txt" ] && { echo "$BASE/jobs_exp22b.txt exists: already submitted, refusing"; exit 1; }
[ -e "$HOME/venvs/fs_hf/requirements.lock.txt" ] || { echo "no ~/venvs/fs_hf/requirements.lock.txt"; exit 1; }
cd "$EXP22_DIR"
git -C "$EXP22_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/git commit: /' || true
mkdir -p logs
mkdir -p expt_v1/results   # the array tasks write here
cp launch/runs/runs_*.txt .
cat runs_feat.txt runs_vis_real.txt runs_vis_shuf.txt > runs_all.txt
wc -l runs_*.txt
USED=$(squeue -r -h -u "$USER" -q gpu | wc -l)
NS=$(( 32 - USED - 1 )); [ "$NS" -gt 25 ] && NS=25
echo "gpu QOS submit slots used: $USED of 32 -> array of $NS tasks"
[ "$NS" -ge 6 ] || { echo "EXP22B NOT SUBMITTED: only $((32 - USED)) gpu QOS submit slots free (need 7); rerun later"; exit 1; }
echo "prep: sbatch --wait on normal, started $(date +%T)"
if ! sbatch --wait --parsable --export=ALL launch/prep_farmshare.sbatch > logs/prep_jobid \
   || ! grep -q "EXP22B PREP DONE" "logs/exp22b_prep_$(cat logs/prep_jobid).out"; then
  tail -25 logs/exp22b_prep_*.out; echo "EXP22B PREP FAILED: nothing submitted"; exit 1
fi
PREP=$(cat logs/prep_jobid)
echo "prep $PREP passed at $(date +%T):"; tail -6 "logs/exp22b_prep_$PREP.out"
N=$(wc -l < runs_all.txt)
PER=$(( (N + NS - 1) / NS ))
WALL=$(( PER * 30 + 30 ))          # minutes: about 17 min per run on an L40S, 30 min allowed per run, 30 spare
H=$(printf '%02d:%02d:00' $((WALL / 60)) $((WALL % 60)))
ARR=$(sbatch --parsable "${GPU[@]}" "${MAIL[@]}" -t "$H" --array=0-$((NS-1))%4 -J exp22b-grid \
      --export=ALL,RUNLIST=runs_all.txt,NSHARDS="$NS" launch/gpu_array.sbatch)
echo "array=$ARR ($N runs in $NS tasks, %4, -t $H each)"
POST=$(sbatch --parsable "${GPU[@]}" "${MAIL[@]}" --dependency=afterany:$ARR --export=ALL launch/post_farmshare.sbatch)
echo "prep=$PREP array=$ARR post=$POST dir=$EXP22_DIR" | tee "$BASE/jobs_exp22b.txt"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R" | head -20
