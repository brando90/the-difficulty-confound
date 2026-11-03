#!/bin/bash
# After the re-grade array: the pre-registered readout from the re-graded results only, packed to g1s3_regrade_results.tgz.
set -uo pipefail
source "$SCRATCH/tdc_g1s3/the-difficulty-confound/experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/seeds3/env.sh" >/dev/null 2>&1
cd "$G1S3" || exit 2
"$PY" "$S3/seeds3_report.py" --out-root "$OUT" --regrade pilot --json g1s3_regrade_report.json --md g1s3_regrade_report.md >g1s3_regrade_report.out 2>&1
A=$(sed -n 's/.*ARRAY=\([0-9]*\).*/\1/p' jobs_g1s3_regrade.txt 2>/dev/null)
[[ -n $A ]] && sacct -j "$A" -o JobID%20,State,ExitCode,Elapsed,NodeList -P >sacct_g1s3_regrade.txt 2>&1
tar czf g1s3_regrade_results.tgz g1s3_regrade_report.json g1s3_regrade_report.md g1s3_regrade_report.out sacct_g1s3_regrade.txt \
  jobs_g1s3_regrade.txt runs/logs/regrade_status.txt $(ls -d runs/runs/*/regrade_pilot) $(ls slurm_logs/g1s3rg_* ) 2>/dev/null
echo "packed $(du -h g1s3_regrade_results.tgz | cut -f1)"
