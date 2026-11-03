#!/bin/bash
# Experiment 23 one-time setup on FarmShare. Run in the JupyterLab terminal (a compute node with python3.12-venv;
# wheat-01 lacks it), from the unpacked package:  bash ~/e23/code/experiments/23_exp13_full_training_8seeds/farmshare/prepare.sh
# 1. builds the venv ($E23_VENV) with the versions the code was tested with, 2. checks the prep sha256s and copies
# the frozen references into $E23_OUT/e13 and $E23_OUT/e17b, 3. runs the CPU tests (about 10-20 min on 2 cores).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
cd "$E23_CODE"
PY312=${PY312:-/usr/bin/python3.12}
if [ ! -x "$E23_PY" ]; then
  "$PY312" -m venv "$E23_VENV"
  "$E23_PY" -m pip install -q --upgrade pip
  # CPU wheel first (small); plain PyPI as the fallback
  "$E23_PY" -m pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    || "$E23_PY" -m pip install -q torch==2.14.1
  "$E23_PY" -m pip install -q numpy==2.5.4 scipy==1.18.1 pytest
  "$E23_PY" -m pip freeze > "$E23_VENV/requirements.lock.txt"
fi
"$E23_PY" -c "import torch, numpy, scipy, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
for exp in e13 e17b; do "$E23_PY" run23.py init --exp $exp --output "$E23_OUT/$exp"; done
"$E23_PY" run23.py count --exp e13
"$E23_PY" run23.py count --exp e17b
# Gate: every test except exact replay. Exact replay of the source runs' first checkpoint (1e-6 bits) was checked on
# the build machine; another CPU may differ by float rounding, so it is reported here, not gated.
"$E23_PY" -m pytest -q -p no:cacheprovider tests -k "not exact_replay"
"$E23_PY" -m pytest -q -p no:cacheprovider tests -k "exact_replay" || echo "NOTE: exact replay differs on this CPU (reported, not a blocker); see above for the size"
echo "PREPARE OK $(date)"
