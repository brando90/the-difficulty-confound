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
# Sherlock is glibc 2.17: pip there may only take manylinux2014/linux_x86_64 wheels, never a source build (the first
# submit, 48476434, built numpy from source and the old assembler rejected endbr64). torch 2.14.1 has no such wheel, so
# the cluster runs torch 2.6.0+cpu / numpy 2.2.6 (PROTOCOL Section 6.2; cross-CPU verify covers the version change).
ml python/3.12.1
VENV=${EXP18_VENV:?set EXP18_VENV}
PIP="$VENV/bin/pip install -q --only-binary=:all: --no-cache-dir"
if [ ! -e "$VENV/.ready" ]; then
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  $PIP --upgrade "pip==25.2"
  $PIP "torch==2.6.0+cpu" --index-url https://download.pytorch.org/whl/cpu || true
  "$VENV/bin/python" -c "import torch" || $PIP --force-reinstall "torch==2.6.0"
  $PIP "numpy==2.2.6" "scipy==1.16.3" "pillow==12.2.0" "pyarrow==20.0.0"
  "$VENV/bin/python" -c "import torch, numpy, scipy, PIL, pyarrow" && touch "$VENV/.ready"
fi
PY=$VENV/bin/python
$PY -c "import torch, numpy, scipy, PIL, pyarrow; print('torch', torch.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'threads', torch.get_num_threads())"
export PATH=$VENV/bin:$PATH
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
