#!/bin/bash
# Submit the whole Experiment 18 grid on Sherlock, once (PROTOCOL Section 7).
# Run from $SCRATCH/exp18r2 after unpacking the package there:  bash exp18_pkg/submit_exp18.sh 2>&1 | tee submit_exp18.log
# r2 (AT-503): the first submit from $SCRATCH/exp18 failed in prep (numpy source build) before any run; that folder is
# left as it is. Prep now runs first, in the foreground on `dev` (starts in seconds; dev refuses sbatch), and nothing is
# submitted unless it passes. The arrays then go to `normal`, and post waits on afterok:prep AND every array ending.
set -euo pipefail
BASE=$SCRATCH/exp18r2
export EXP18_VENV=$SCRATCH/exp18_venv
[ -e "$BASE/jobs_exp18.txt" ] && { echo "jobs_exp18.txt exists: already submitted, refusing"; exit 1; }
cd "$BASE/exp18_pkg"
sha256sum -c --quiet SHA256SUMS && echo "package verified: $(cat FREEZE)"
export EXP18_DIR=$BASE/exp18_pkg/the-difficulty-confound/experiments/18_vision_cnn_complexity_vs_alignment
cd "$EXP18_DIR"
mkdir -p logs
# the run lists were written at the freeze commit (launch/runs/, checked by SHA256SUMS)
cp launch/runs/runs_*.txt .
wc -l runs_*.txt
ml python/3.12.1
rm -f logs/prep_jobid
echo "prep: srun on dev, started $(date +%T)"
if ! srun -p dev -c 2 --mem=8G -t 02:00:00 -J exp18-prep --export=ALL,EXP18_DIR="$EXP18_DIR" \
     bash -c 'echo "$SLURM_JOB_ID" > logs/prep_jobid; exec bash launch/prep_sherlock.sbatch' > logs/exp18_prep.out 2>&1 \
   || ! grep -q "EXP18 PREP DONE" logs/exp18_prep.out; then
  tail -25 logs/exp18_prep.out; echo "EXP18 PREP FAILED: nothing submitted"; exit 1
fi
PREP=$(cat logs/prep_jobid)
echo "prep $PREP passed at $(date +%T):"; tail -6 logs/exp18_prep.out
IDS=""
for ph in mnist_main mnist_H fashion_main kmnist_main fashion_H kmnist_H mis cifar10_main cifar10_H; do
  N=$(wc -l < "runs_$ph.txt")
  case $ph in cifar10_*) T=08:00:00; M=6G ;; *) T=03:00:00; M=4G ;; esac
  J=$(sbatch --parsable -t "$T" --mem=$M --array=0-$((N-1))%150 \
      --export=ALL,RUNLIST="runs_$ph.txt",NSHARDS="$N",VISION_DATA="$SCRATCH/exp18_npz" -J "exp18-$ph" launch/sherlock_cpu.sbatch)
  echo "$ph=$J ($N runs, -t $T)"
  IDS="$IDS:$J"
done
# post waits for the successful prep AND for every array to end (some failed tasks still get packed and reported)
POST=$(sbatch --parsable --dependency=afterok:$PREP,afterany$IDS --export=ALL,EXP18_DIR="$EXP18_DIR" launch/post_sherlock.sbatch)
echo "prep=$PREP post=$POST arrays=${IDS#:}" | tee "$BASE/jobs_exp18.txt"
scontrol show job "$POST" | grep -o 'Dependency=[^ ]*'
squeue -u "$USER" -o "%.18i %.12P %.20j %.8T %.10M %.6D %R" | head -20
