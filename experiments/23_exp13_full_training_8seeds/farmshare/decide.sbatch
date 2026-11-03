#!/bin/bash
# The stopping rule after a stage (PROTOCOL Section 3). Args: EXP STAGE. Exit 1 = runs missing: the next stage's
# arrays are cancelled (--kill-on-invalid-dep) and `submit_full.sh --from-stage STAGE` resubmits (resume-safe).
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
set -euo pipefail
source "$HOME/e23/code/experiments/23_exp13_full_training_8seeds/farmshare/env.sh"
cd "$E23_CODE"
"$E23_PY" -u run23.py decide --exp "$1" --stage "$2" --output "$E23_OUT/$1"
