#!/bin/bash
#SBATCH -J exp22
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/exp22_%x_%A_%a.out
# One array task = one run of a run list (--array=0-(N-1), NSHARDS=N). Results land in expt_v1/results/<phase>/;
# a finished run is skipped on resubmission. Run from experiments/22_shared_structure_real_data.
set -euo pipefail
export EXP22_DATA=$SCRATCH/exp22_data OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
cd expt_v1
"${EXP22_VENV:?set EXP22_VENV}/bin/python" run.py --runlist "../$RUNLIST" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
