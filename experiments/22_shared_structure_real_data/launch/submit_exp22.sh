#!/bin/bash
# Submit the whole Experiment 22 grid on Sherlock, once (PROTOCOL Section 6).
# Upload exp22_pkg.tar.xz to $SCRATCH/exp22, then:
#   cd $SCRATCH/exp22 && tar xJf exp22_pkg.tar.xz && bash exp22_pkg/submit_exp22.sh 2>&1 | tee submit_exp22.log
# Prep runs first in the foreground on `dev` (dev refuses sbatch); nothing is submitted unless it prints EXP22 PREP
# DONE. Then one GPU array per phase on `gpu` (V100-class by default: GPU_CON=GPU_GEN:VLT; GPU_CON= for any GPU),
# and a post job that waits for every array to end. Slurm mail END,FAIL goes to both of Brando's addresses.
set -euo pipefail
BASE=$SCRATCH/exp22
MAIL=(--mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com)
export EXP22_VENV=$SCRATCH/exp22_venv
GPU_CON=${GPU_CON-GPU_GEN:VLT}
[ -e "$BASE/jobs_exp22.txt" ] && { echo "jobs_exp22.txt exists: already submitted, refusing"; exit 1; }
cd "$BASE/exp22_pkg"
sha256sum -c --quiet SHA256SUMS && echo "package verified: $(cat FREEZE)"
export EXP22_DIR=$BASE/exp22_pkg/the-difficulty-confound/experiments/22_shared_structure_real_data
cd "$EXP22_DIR"
mkdir -p logs
cp launch/runs/runs_*.txt .
wc -l runs_*.txt
ml python/3.12.1
echo "prep: srun on dev, started $(date +%T)"
if ! srun -p dev -c 4 --mem=16G -t 02:00:00 -J exp22-prep "${MAIL[@]}" --export=ALL,EXP22_DIR="$EXP22_DIR" \
     bash launch/prep_sherlock.sh > logs/exp22_prep.out 2>&1 || ! grep -q "EXP22 PREP DONE" logs/exp22_prep.out; then
  tail -25 logs/exp22_prep.out; echo "EXP22 PREP FAILED: nothing submitted"; exit 1
fi
echo "prep passed at $(date +%T):"; tail -6 logs/exp22_prep.out
CON=(); [ -n "$GPU_CON" ] && CON=(-C "$GPU_CON")
IDS=""
for ph in feat vis_real vis_shuf txt_real txt_shuf; do
  N=$(wc -l < "runs_$ph.txt")
  case $ph in vis_*|feat) T=03:00:00 ;; *) T=01:30:00 ;; esac
  J=$(sbatch --parsable -t "$T" "${CON[@]}" "${MAIL[@]}" --array=0-$((N-1))%30 -J "exp22-$ph" \
      --export=ALL,RUNLIST="runs_$ph.txt",NSHARDS="$N" launch/gpu_array.sbatch)
  echo "$ph=$J ($N runs, -t $T, constraint ${GPU_CON:-none})"
  IDS="$IDS:$J"
done
POST=$(sbatch --parsable "${CON[@]}" "${MAIL[@]}" --dependency=afterany$IDS --export=ALL,EXP22_DIR="$EXP22_DIR" launch/post_sherlock.sbatch)
echo "post=$POST arrays=${IDS#:} constraint=${GPU_CON:-none}" | tee "$BASE/jobs_exp22.txt"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R" | head -20
