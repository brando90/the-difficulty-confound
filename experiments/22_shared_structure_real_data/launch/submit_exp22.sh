#!/bin/bash
# Submit the whole Experiment 22 grid on FarmShare, once (PROTOCOL Section 9, Amendment 1). From the JupyterLab
# terminal (a compute node with sbatch and internet), with the experiment at ~/exp22/... from a git clone of the frozen
# commit or the unpacked package:
#   bash <experiment folder>/launch/submit_exp22.sh 2>&1 | tee ~/exp22/submit_exp22.log
# 1) prep: `sbatch --wait` on -p normal (venv ~/venvs/exp22 from ~/venvs/fs_hf/requirements.lock.txt, data under
#    ~/exp22, every hash checked); nothing else is submitted unless it prints EXP22 PREP DONE.
# 2) one GPU array of all 155 runs on -p gpu --qos=gpu --gres=gpu:1 (L40S, bf16), throttled %4: never more than
#    4 GPUs at once (FarmShare's per-user limit); every task's wall time is under the 2-day cap.
# 3) a post job (GPU) after the array ends: verify.py --stage remote, then ~/exp22/exp22_results.tgz.
# Slurm mail END,FAIL to both of Brando's addresses on every job (one mail per array, never per task).
set -euo pipefail
BASE=$HOME/exp22
MAIL=(--mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com)
GPU=(-p gpu --qos=gpu --gres=gpu:1)
export EXP22_VENV=$HOME/venvs/exp22
export EXP22_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$BASE"
[ -e "$BASE/jobs_exp22.txt" ] && { echo "$BASE/jobs_exp22.txt exists: already submitted, refusing"; exit 1; }
[ -e "$HOME/venvs/fs_hf/requirements.lock.txt" ] || { echo "no ~/venvs/fs_hf/requirements.lock.txt"; exit 1; }
cd "$EXP22_DIR"
git -C "$EXP22_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/git commit: /' || true
mkdir -p logs
cp launch/runs/runs_*.txt .
cat runs_feat.txt runs_vis_real.txt runs_vis_shuf.txt runs_txt_real.txt runs_txt_shuf.txt > runs_all.txt
wc -l runs_*.txt
echo "prep: sbatch --wait on normal, started $(date +%T)"
if ! sbatch --wait --parsable --export=ALL launch/prep_farmshare.sbatch > logs/prep_jobid \
   || ! grep -q "EXP22 PREP DONE" "logs/exp22_prep_$(cat logs/prep_jobid).out"; then
  tail -25 logs/exp22_prep_*.out; echo "EXP22 PREP FAILED: nothing submitted"; exit 1
fi
PREP=$(cat logs/prep_jobid)
echo "prep $PREP passed at $(date +%T):"; tail -6 "logs/exp22_prep_$PREP.out"
N=$(wc -l < runs_all.txt)
ARR=$(sbatch --parsable "${GPU[@]}" "${MAIL[@]}" -t 03:00:00 --array=0-$((N-1))%4 -J exp22-grid \
      --export=ALL,RUNLIST=runs_all.txt,NSHARDS="$N" launch/gpu_array.sbatch)
echo "array=$ARR ($N runs, %4, -t 03:00:00 each)"
POST=$(sbatch --parsable "${GPU[@]}" "${MAIL[@]}" --dependency=afterany:$ARR --export=ALL launch/post_farmshare.sbatch)
echo "prep=$PREP array=$ARR post=$POST dir=$EXP22_DIR" | tee "$BASE/jobs_exp22.txt"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R" | head -20
