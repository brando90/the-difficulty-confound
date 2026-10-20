#!/bin/bash
# Environment + data for Experiment 22 on Sherlock (PROTOCOL Section 6). Run by submit_exp22.sh as a foreground
# `srun -p dev`; prints "EXP22 PREP DONE" only if the venv imports, the raw and prepared data hash to
# expt_v1/data_sha256.json and the split tests pass. Nothing is submitted otherwise.
set -euo pipefail
cd "${EXP22_DIR:?set EXP22_DIR}"
{ ml python/3.12.1 || module load python/3.12.1; } 2>/dev/null || true
VENV=${EXP22_VENV:?set EXP22_VENV}
# Sherlock is glibc 2.17: wheels only (a numpy source build broke Exp 18's first prep). torch 2.6.0 from PyPI bundles
# CUDA 12.4, which the GPU nodes' driver runs.
PIP="$VENV/bin/pip install -q --only-binary=:all: --no-cache-dir"
if [ ! -e "$VENV/.ready" ]; then
  rm -rf "$VENV"
  "$(command -v python3.12 || command -v python3)" -m venv "$VENV"
  $PIP --upgrade "pip==25.2"
  $PIP "torch==2.6.0" "numpy==2.2.6" "scipy==1.16.3" "pillow==12.2.0" "pyarrow==20.0.0" "scikit-learn==1.7.2"
  "$VENV/bin/python" -c "import torch, numpy, scipy, PIL, pyarrow, sklearn" && touch "$VENV/.ready"
fi
PY=$VENV/bin/python
$PY -c "import torch, numpy, scipy, sklearn; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__)"
EXP22_RAW=$SCRATCH/exp22_raw bash launch/fetch_data.sh
cd expt_v1
EXP22_DATA=$SCRATCH/exp22_data $PY prep.py --raw "$SCRATCH/exp22_raw"
EXP22_DATA=$SCRATCH/exp22_data $PY test_splits.py
echo "EXP22 PREP DONE"
