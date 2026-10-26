#!/bin/bash
# Write the seeds3 report and pack $SCRATCH/tdc_g1s3/g1s3_results.tgz (per-item metrics, generations, logs; no weights).
# Idempotent: every array task runs it at its end, and the report job runs it once after the array.
set -uo pipefail
source "$SCRATCH/tdc_g1s3/the-difficulty-confound/experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/seeds3/env.sh" >/dev/null 2>&1
cd "$G1S3" || exit 2
"$PY" "$S3/seeds3_report.py" --out-root "$OUT" --json g1s3_report.json --md g1s3_report.md >g1s3_report.out 2>&1
A=$(sed -n 's/.*ARRAY=\([0-9]*\).*/\1/p' jobs_g1s3.txt 2>/dev/null)
[[ -n $A ]] && sacct -j "$A" -o JobID%20,State,ExitCode,Elapsed,NodeList -P >sacct_g1s3.txt 2>&1
TMP=$(mktemp -p "$G1S3" g1s3_results.XXXX.tgz)
tar czf "$TMP" --exclude='runs/runs/*/final' g1s3_report.json g1s3_report.md g1s3_report.out sacct_g1s3.txt \
  jobs_g1s3.txt inventory_g1s3.txt submit_g1s3.log runs/logs runs/runs slurm_logs 2>/dev/null
mv -f "$TMP" "$G1S3/g1s3_results.tgz"
echo "$(date '+%m-%d-%Y %H:%M:%S') packed $(du -h g1s3_results.tgz | cut -f1)"
