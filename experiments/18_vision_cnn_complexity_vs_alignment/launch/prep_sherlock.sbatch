#!/bin/bash
#SBATCH -J exp18-prep
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o logs/exp18_prep_%j.out
# Environment + data for Experiment 18 on Sherlock (PROTOCOL Section 7). Fails (so no array starts) unless the raw
# files and the decoded arrays hash to expt_v1/data_sha256.json.
set -euo pipefail
cd "$EXP18_DIR"
ml python/3.12.1 2>/dev/null || ml python/3.9.0
if [ ! -x "$HOME/venvs/exp18/bin/python" ]; then
  python3 -m venv "$HOME/venvs/exp18"
  "$HOME/venvs/exp18/bin/pip" install -q --upgrade pip
  "$HOME/venvs/exp18/bin/pip" install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu \
    || "$HOME/venvs/exp18/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
  "$HOME/venvs/exp18/bin/pip" install -q "numpy==2.5.4" scipy matplotlib pyarrow pillow \
    || "$HOME/venvs/exp18/bin/pip" install -q numpy scipy matplotlib pyarrow pillow
fi
PY=$HOME/venvs/exp18/bin/python
$PY -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
export PATH=$HOME/venvs/exp18/bin:$PATH
VISION_RAW=$SCRATCH/exp18_raw VISION_DATA=$SCRATCH/exp18_npz bash launch/fetch_data.sh
VISION_DATA=$SCRATCH/exp18_npz $PY - <<'PYEOF'
import hashlib, json, os, sys
sys.path.insert(0, 'expt_v1')
import numpy as np
import data as D
want = json.load(open('expt_v1/data_sha256.json'))
raw = os.environ.get('SCRATCH') + '/exp18_raw'
bad = []
for f, h in want['raw'].items():
    if hashlib.sha256(open(os.path.join(raw, f), 'rb').read()).hexdigest() != h:
        bad.append(f)
for ds, arrs in want['arrays'].items():
    d = D.load(ds)
    for k, h in arrs.items():
        if hashlib.sha256(np.ascontiguousarray(d[k]).tobytes()).hexdigest() != h:
            bad.append(ds + '/' + k)
print('data check:', 'OK' if not bad else 'MISMATCH ' + ' '.join(bad))
sys.exit(1 if bad else 0)
PYEOF
cd expt_v1 && VISION_DATA=$SCRATCH/exp18_npz $PY test_tasks.py
echo "EXP18 PREP DONE"
