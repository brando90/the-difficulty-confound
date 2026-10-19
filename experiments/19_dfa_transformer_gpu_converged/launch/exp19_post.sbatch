#!/bin/bash
# Experiment 19 analysis on a CPU node after training: GPU hours, analyze.py stages 1-2, summary.py, verify.py,
# then two archives to fetch: exp19_small.tgz (numbers) and exp19_full.tgz (adds predictions and the final checkpoint).
#SBATCH --job-name=tdc_exp19_post
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
set -euo pipefail
: "${SCRATCH:?}" "${BASE:?}" "${X19:?}" "${TRAIN_JOB:?}"
source "$X19/launch/env.sh"
export OMP_NUM_THREADS=1
cd "$X19/expt_v1"
"$PY" "$X19/launch/budget.py" --job "$TRAIN_JOB" >"$OUT/gpu_usage.json"
cat "$OUT/gpu_usage.json"
if [[ -f $OUT/NOT_RUN.txt || ! -f $OUT/run_state.json ]]; then
  echo "training did not run or did not finish a checkpoint"; tar czf "$BASE/exp19_small.tgz" -C "$BASE" exp19_results/gpu_usage.json $(cd "$BASE" && ls exp19_results/*.txt exp19_results/*.json exp19_results/*.log 2>/dev/null) slurm_logs; exit 0
fi
[[ -f $OUT/manifest.json ]] || echo "WARNING: no manifest.json (training stopped by the time limit, not by the budget rule)"
"$PY" analyze.py --stage 1 --results "$OUT" | tee "$OUT/stage1.log"
"$PY" analyze.py --stage 2 --results "$OUT" >"$OUT/stage2.log" 2>&1 || { tail -30 "$OUT/stage2.log"; }
"$PY" summary.py --results "$OUT" | tee "$OUT/summary.log" || true
"$PY" verify.py --results "$OUT" --replay all >"$OUT/verify.log" 2>&1 || true
tail -30 "$OUT/verify.log"
cd "$BASE"
SMALL=(exp19_results/metrics.csv exp19_results/run_state.json exp19_results/budget_at_start.json exp19_results/gpu_usage.json exp19_results/gpu.txt)
for f in manifest.json fits.json forecasts_frozen.json analysis.json tstar.csv summary.json convergence_runs.json verification.json train.log stage1.log stage2.log summary.log verify.log; do
  [[ -f exp19_results/$f ]] && SMALL+=(exp19_results/$f)
done
tar czf exp19_small.tgz "${SMALL[@]}" slurm_logs
TOP=$("$PY" -c "import json; print(json.load(open('exp19_results/run_state.json'))['T_done'][-1])")
tar czf exp19_full.tgz "${SMALL[@]}" exp19_results/predictions "exp19_results/checkpoints/T_${TOP}.npz"
ls -la exp19_small.tgz exp19_full.tgz
echo "EXP19 POST DONE"
