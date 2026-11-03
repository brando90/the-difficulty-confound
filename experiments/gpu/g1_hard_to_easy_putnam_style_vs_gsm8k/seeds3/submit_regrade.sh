#!/bin/bash
# Submit the re-grade array (16 tasks) and its report job from $SCRATCH/tdc_g1s3. Ids go to jobs_g1s3_regrade.txt.
set -uo pipefail
source "$SCRATCH/tdc_g1s3/the-difficulty-confound/experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/seeds3/env.sh" >/dev/null 2>&1
cd "$G1S3" || exit 2
mkdir -p slurm_logs
[[ $(ls -d "$OUT"/runs/qwen2.5-0.5b__* | wc -l) == 16 ]] || { echo "expected 16 run dirs" >&2; exit 2; }
for d in "$OUT"/runs/qwen2.5-0.5b__*; do
  [[ $d == *__base ]] || [[ -f $d/final/config.json ]] || { echo "no final weights in $d" >&2; exit 2; }
done
A=$(sbatch --parsable "$S3/regrade.sbatch") || exit 2
R=$(sbatch --parsable --dependency=afterany:"$A" --partition=normal --time=00:20:00 --job-name=g1s3rg_rep \
  --mail-type=BEGIN,END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com \
  --output=slurm_logs/%x_%j.out --error=slurm_logs/%x_%j.err --wrap="bash $S3/regrade_report.sh") || exit 2
echo "ARRAY=$A REPORT=$R submitted $(date '+%m-%d-%Y %H:%M:%S')" | tee jobs_g1s3_regrade.txt
