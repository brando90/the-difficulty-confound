#!/bin/bash
#SBATCH -J exp22-prep
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/exp22_prep_%j.out
# Environment + data for Experiment 22 on FarmShare (PROTOCOL Section 9, Amendment 1). submit_exp22.sh runs it with
# `sbatch --wait` and submits nothing unless it prints "EXP22 PREP DONE": the venv imports, the raw and prepared data
# hash to expt_v1/data_sha256.json, and the split tests pass. Everything lives under $HOME (FarmShare has no /scratch).
set -euo pipefail
cd "${EXP22_DIR:?set EXP22_DIR}"
VENV=${EXP22_VENV:?set EXP22_VENV}
LOCK=$HOME/venvs/fs_hf/requirements.lock.txt
if [ ! -e "$VENV/.ready" ]; then
  rm -rf "$VENV"
  /usr/bin/python3.12 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip wheel
  # the shared fs_hf stack exactly as locked (torch 2.13.0 + CUDA 13 wheels), then this experiment's extras,
  # constrained to the lock so nothing in it moves
  "$VENV/bin/pip" install -q -r "$LOCK"
  "$VENV/bin/pip" install -q -c "$LOCK" scikit-learn scipy pyarrow pillow
  "$VENV/bin/python" -c "import torch, numpy, scipy, PIL, pyarrow, sklearn" && touch "$VENV/.ready"
fi
"$VENV/bin/pip" freeze > logs/exp22_venv_freeze.txt
PY=$VENV/bin/python
$PY -c "import torch, numpy, scipy, sklearn; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__)"
EXP22_RAW=$HOME/exp22/raw bash launch/fetch_data.sh
cd expt_v1
EXP22_DATA=$HOME/exp22/data $PY prep.py --raw "$HOME/exp22/raw"
EXP22_DATA=$HOME/exp22/data $PY test_splits.py
echo "EXP22 PREP DONE"
