#!/bin/bash
# AT-494 on Sherlock, from a login shell, after unpacking diag_cosine_lr into the Experiment 19 package:
#   cd $SCRATCH/tdc_gpu && bash $X19/diag_cosine_lr/submit_diag.sh 2>&1 | tee submit_diag494.log
# One ordinary batch job on one GPU and nothing else. Uses Experiment 19's venv (torch 2.6.0+cu124).
set -uo pipefail
BASE=$SCRATCH/tdc_gpu
X19=$BASE/tdc_exp19/the-difficulty-confound/experiments/19_dfa_transformer_gpu_converged
cd "$BASE" || exit 2
mkdir -p slurm_logs
(cd "$X19/diag_cosine_lr" && sha256sum -c --quiet SHA256SUMS) || { echo "diag files do not match SHA256SUMS"; exit 2; }
(cd "$X19/expt_v1" && echo "d155c84f7f7a2c5794db45d10797af2a925c23edcdf5331296ee64a8b95e123f  common.py
757a1f0e040f4d9501de31fa3e3d91253976caea5c366f891faa0c520e5c1274  ensemble.py
711142e896d8362af3e33a56086cc1967a304995f8eb82c02c34c6befac12a79  gpu_train.py" | sha256sum -c --quiet) || { echo "expt_v1 differs from Exp 19's run"; exit 2; }
[[ -e $BASE/diag494_results || -f $BASE/jobs_diag494.txt ]] && { echo "AT-494 was already submitted"; exit 2; }
[[ -f $BASE/venv_exp19/.ready && -f $BASE/exp19_testmask.npz ]] || { echo "venv_exp19 or exp19_testmask.npz missing"; exit 2; }
CC=$(sh_node_feat -p gpu 2>/dev/null | grep -o 'GPU_CC:[0-9.]*' | sort -u | awk -F: '$2>=7.0{print $0}' | paste -sd'|')
CON=()
[[ -n $CC ]] && CON=(-C "$CC")
J=$(sbatch --parsable "${CON[@]}" --export=ALL,BASE="$BASE",X19="$X19",VENV="$BASE/venv_exp19" \
  "$X19/diag_cosine_lr/diag.sbatch") || { echo "submit failed"; exit 2; }
echo "diag494=$J constraint=${CC:-none}" | tee "$BASE/jobs_diag494.txt"
squeue -u "$USER" -o "%.18i %.14j %.9P %.8T %.10M %.30R"
echo "When it ends: $BASE/diag494_results.tgz; log slurm_logs/tdc_diag494_$J.out"
