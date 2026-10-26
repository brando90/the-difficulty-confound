#!/bin/bash
#SBATCH -J exp22b
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/exp22b_%A_%a.out
# One array task = one run of runs_all.txt (submit_exp22b.sh: --array=0-(NS-1)%4, so at most 4 GPUs at once).
# Results land in expt_v1/results/<phase>/; a finished run is skipped on resubmission.
# Run from experiments/22_shared_structure_real_data.
set -euo pipefail
export EXP22_DATA=$HOME/exp22/data OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
cd expt_v1
"${EXP22_VENV:?set EXP22_VENV}/bin/python" run.py --runlist "../$RUNLIST" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
