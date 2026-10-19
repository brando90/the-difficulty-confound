#!/bin/bash
# Not needed: prep_sherlock.sbatch builds the venv ($EXP18_VENV) from wheels only. Kept for a manual login-node build:
#   EXP18_VENV=$SCRATCH/exp18_venv bash launch/setup_sherlock.sh
set -euo pipefail
ml python/3.12.1
VENV=${EXP18_VENV:?set EXP18_VENV}
PIP="$VENV/bin/pip install -q --only-binary=:all: --no-cache-dir"
python3 -m venv "$VENV"
$PIP --upgrade "pip==25.2"
$PIP "torch==2.6.0+cpu" --index-url https://download.pytorch.org/whl/cpu || true
"$VENV/bin/python" -c "import torch" || $PIP --force-reinstall "torch==2.6.0"
$PIP "numpy==2.2.6" "scipy==1.16.3" "pillow==12.2.0" "pyarrow==20.0.0"
"$VENV/bin/python" -c "import torch, numpy, scipy, PIL, pyarrow; print('torch', torch.__version__)" && touch "$VENV/.ready"
