#!/bin/bash
#SBATCH -J exp18-post
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 10:00:00
#SBATCH -o logs/exp18_post_%j.out
# After every phase array: verify.py --stage remote (exact checks on this CPU model) and pack results/ for transfer.
set -uo pipefail
cd "$EXP18_DIR"
export VISION_DATA=$SCRATCH/exp18_npz OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CPU_LOCK_DIR=$TMPDIR
PY=$HOME/venvs/exp18/bin/python
$PY - <<'PYEOF'
import json, os, sys
sys.path.insert(0, 'expt_v1')
import run
for ph in run.ORDER:
    specs = [json.loads(l) for l in open(f'runs_{ph}.txt')]
    print(f'phase {ph}: {sum(os.path.exists(run.out_path(s)) for s in specs)} of {len(specs)} runs present')
PYEOF
ls expt_v1/results/*/*.error 2>/dev/null | head -20
(cd expt_v1 && $PY verify.py --stage remote --workers 8 --retrain 4) 2>&1 | tail -40
cd expt_v1
tar czf "$SCRATCH/exp18/exp18_results.tgz" results
sha256sum "$SCRATCH/exp18/exp18_results.tgz" | tee "$SCRATCH/exp18/exp18_results.tgz.sha256"
ls -la "$SCRATCH/exp18/exp18_results.tgz"
echo "EXP18 POST DONE"
