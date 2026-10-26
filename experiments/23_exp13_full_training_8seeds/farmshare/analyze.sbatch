#!/bin/bash
# After both experiments stopped: aggregate, the frozen analyses, verification, RESULTS.md. Submitted by submit_full.sh.
# Small outputs are copied to $E23_CODE/results/ for the commit; checkpoints and training state stay in $E23_OUT.
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
set -euo pipefail
source "$HOME/e23/code/experiments/23_exp13_full_training_8seeds/farmshare/env.sh"
cd "$E23_CODE"
for exp in e13 e17b; do "$E23_PY" -u run23.py aggregate --exp $exp --output "$E23_OUT/$exp"; done
"$E23_PY" -u analyze23.py --exp e17b --output "$E23_OUT/e17b"
"$E23_PY" -u analyze23.py --exp e13 --output "$E23_OUT/e13"
"$E23_PY" -u verify23.py --out "$E23_OUT" || echo "VERIFY FAILED (see verify_log.json); RESULTS.md still written"
"$E23_PY" -u report23.py --out "$E23_OUT" --results "$E23_CODE/RESULTS.md"
mkdir -p "$E23_CODE/results"
for exp in e13 e17b; do
  mkdir -p "$E23_CODE/results/$exp"
  (cd "$E23_OUT/$exp" && cp -r --parents STOP decisions manifest.json metrics.csv cooldown.csv val.csv mixes.json \
     $(ls gates.json analysis.json fits.json forecasts.csv eps_star_rule.json eps_1.10/gates.json eps_1.10/analysis.json \
          distractor_loss.csv analysis_e17b.json pool_loss.csv 2>/dev/null) "$E23_CODE/results/$exp/")
done
cp "$E23_OUT/verify_log.json" "$E23_CODE/results/" 2>/dev/null || true
echo "ANALYZE DONE $(date): $E23_CODE/RESULTS.md"
