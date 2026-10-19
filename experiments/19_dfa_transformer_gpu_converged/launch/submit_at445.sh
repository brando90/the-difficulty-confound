#!/bin/bash
# AT-445 / Experiment 19 on Sherlock. From a Sherlock login shell, after unpacking the upload into $SCRATCH/tdc_gpu:
#   cd $SCRATCH/tdc_gpu && bash tdc_exp19/submit_at445.sh 2>&1 | tee submit_at445.log
# Submits two ordinary batch jobs and nothing else: training on one GPU (starts after AT-441's pilot array, if one
# was submitted, and sizes itself to the GPU budget left), then analysis on a CPU node. Nothing persistent.
set -uo pipefail
BASE=$SCRATCH/tdc_gpu
cd "$BASE" || exit 2
mkdir -p slurm_logs
X19=$BASE/tdc_exp19/the-difficulty-confound/experiments/19_dfa_transformer_gpu_converged
(cd "$BASE/tdc_exp19" && sha256sum -c --quiet SHA256SUMS) || { echo "package files do not match SHA256SUMS"; exit 2; }
echo "package verified: $(cat "$BASE/tdc_exp19/FREEZE")"
[[ -e $BASE/exp19_results || -f $BASE/jobs_at445.txt ]] && { echo "Experiment 19 was already submitted (jobs_at445.txt / exp19_results exist)"; exit 2; }
DEP=()
if [[ -f jobs_at441.txt ]]; then
  A=$(grep -o 'pilot=[0-9]*' jobs_at441.txt | cut -d= -f2)
  if [[ -n $A ]] && squeue -h -j "$A" -o '%T %R' 2>/dev/null | grep -q .; then
    if squeue -h -j "$A" -o '%R' | grep -q DependencyNeverSatisfied; then
      echo "AT-441 pilot $A can never start (its prep failed); not waiting for it"
    else
      DEP=(--dependency=afterany:$A)
    fi
  fi
fi
CC=$(sh_node_feat -p gpu 2>/dev/null | grep -o 'GPU_CC:[0-9.]*' | sort -u | awk -F: '$2>=7.0{print $0}' | paste -sd'|')
CON=()
[[ -n $CC ]] && CON=(-C "$CC")
T=$(sbatch --parsable "${DEP[@]}" "${CON[@]}" --export=ALL,BASE="$BASE",X19="$X19" "$X19/launch/exp19_train.sbatch") || { echo "train submit failed"; exit 2; }
P=$(sbatch --parsable --dependency=afterany:"$T" --export=ALL,BASE="$BASE",X19="$X19",TRAIN_JOB="$T" "$X19/launch/exp19_post.sbatch") || { echo "post submit failed"; exit 2; }
echo "train=$T post=$P constraint=${CC:-none} after=${DEP[*]:-none}" | tee "$BASE/jobs_at445.txt"
squeue -u "$USER" -o "%.18i %.14j %.9P %.8T %.10M %.30R"
echo "When post finishes: $BASE/exp19_small.tgz and $BASE/exp19_full.tgz; log slurm_logs/tdc_exp19_post_$P.out"
