#!/bin/bash
# Submit the whole Experiment 18 grid on Sherlock's normal CPU partition, once (PROTOCOL Section 7).
# Run from $SCRATCH/exp18 after unpacking the package:  bash exp18_pkg/submit_exp18.sh 2>&1 | tee submit_exp18.log
set -euo pipefail
BASE=$SCRATCH/exp18
[ -e "$BASE/jobs_exp18.txt" ] && { echo "jobs_exp18.txt exists: already submitted, refusing"; exit 1; }
cd "$BASE/exp18_pkg"
sha256sum -c --quiet SHA256SUMS && echo "package verified: $(cat FREEZE)"
export EXP18_DIR=$BASE/exp18_pkg/the-difficulty-confound/experiments/18_vision_cnn_complexity_vs_alignment
cd "$EXP18_DIR"
mkdir -p logs
# the run lists were written at the freeze commit (launch/runs/, checked by SHA256SUMS)
cp launch/runs/runs_*.txt .
wc -l runs_*.txt
PREP=$(sbatch --parsable --export=ALL,EXP18_DIR="$EXP18_DIR" launch/prep_sherlock.sbatch)
IDS=""
for ph in mnist_main mnist_H fashion_main kmnist_main fashion_H kmnist_H mis cifar10_main cifar10_H; do
  N=$(wc -l < "runs_$ph.txt")
  case $ph in cifar10_*) T=08:00:00 ;; *) T=03:00:00 ;; esac
  J=$(sbatch --parsable -t "$T" --mem=4G --dependency=afterok:$PREP --array=0-$((N-1))%150 \
      --export=ALL,RUNLIST="runs_$ph.txt",NSHARDS="$N",VISION_DATA="$SCRATCH/exp18_npz" -J "exp18-$ph" launch/sherlock_cpu.sbatch)
  echo "$ph=$J ($N runs, -t $T)"
  IDS="$IDS:$J"
done
POST=$(sbatch --parsable --dependency=afterany$IDS --export=ALL,EXP18_DIR="$EXP18_DIR" launch/post_sherlock.sbatch)
echo "prep=$PREP post=$POST arrays=${IDS#:}" | tee "$BASE/jobs_exp18.txt"
squeue -u "$USER" -o "%.18i %.12P %.20j %.8T %.10M %.6D %R" | head -20
