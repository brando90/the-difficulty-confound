#!/bin/bash
#SBATCH -J exp18-vision
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 02:00:00
#SBATCH -o logs/exp18_%A_%a.out
# One Slurm array task = one shard of a run list (one run per task when --array=0-(N-1) and NSHARDS=N).
# Usage, from experiments/18_vision_cnn_complexity_vs_alignment:
#   mkdir -p logs && $HOME/venvs/exp18/bin/python expt_v1/run.py --list cifar10_main > runs_cifar10_main.txt
#   N=$(wc -l < runs_cifar10_main.txt); sbatch --array=0-$((N-1))%200 --export=ALL,RUNLIST=runs_cifar10_main.txt,NSHARDS=$N launch/sherlock_cpu.sbatch
# Results land in expt_v1/results/ exactly as local runs do (finished runs are skipped on resubmission).
set -euo pipefail
export VISION_DATA=${VISION_DATA:-$SCRATCH/exp18_npz}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CPU_LOCK_DIR=$TMPDIR
$HOME/venvs/exp18/bin/python expt_v1/run.py --runlist "$RUNLIST" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
