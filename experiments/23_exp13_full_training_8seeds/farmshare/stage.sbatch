#!/bin/bash
# One array task of one stage: 4 runs in parallel (one thread each). Args: EXP STAGE. Submitted by submit_full.sh,
# which sets the job name, array range with %throttle, time limit, dependencies, mail and log path.
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=6G
set -euo pipefail
source "$HOME/e23/code/experiments/23_exp13_full_training_8seeds/farmshare/env.sh"
cd "$E23_CODE"
EXP=$1; STAGE=$2
echo "$EXP stage $STAGE task $SLURM_ARRAY_TASK_ID on $(hostname) $(date)"
"$E23_PY" -u run23.py stage --exp "$EXP" --stage "$STAGE" --task "$SLURM_ARRAY_TASK_ID" --output "$E23_OUT/$EXP" --workers 4
echo "done $(date)"
