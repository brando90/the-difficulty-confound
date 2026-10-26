#!/bin/bash
# Experiment 23 CPU smoke: 2 cells (Exp 13: seed 13101, n = 32 target 1, alpha 0.4; Exp 17b: seed 17101, cov32 pool,
# alpha 0, scored on all 4 targets), real stage 0 (2^20 tokens + cooldown), in a scratch output folder.
# Prints training throughput on this node and the compute estimate. Submit: sbatch farmshare/smoke.sbatch (from $E23_CODE)
#SBATCH -J e23_smoke
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o %x_%j.out
set -euo pipefail
source "$HOME/e23/code/experiments/23_exp13_full_training_8seeds/farmshare/env.sh"
cd "$E23_CODE"
echo "host $(hostname) start $(date)"; lscpu | grep -E "Model name|^CPU\(s\)" || true
rm -rf "$E23_HOME/smoke"
"$E23_PY" -u run23.py smoke --output "$E23_HOME/smoke"
echo "SMOKE OK $(date)"
