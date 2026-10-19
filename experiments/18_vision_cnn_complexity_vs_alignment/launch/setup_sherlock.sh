#!/bin/bash
# One-time environment on Sherlock (login node is fine; ~5 min). Creates $HOME/venvs/exp18 with CPU torch.
set -euo pipefail
ml python/3.12.1 2>/dev/null || ml python/3.9.0
python3 -m venv "$HOME/venvs/exp18"
"$HOME/venvs/exp18/bin/pip" install -q --upgrade pip
"$HOME/venvs/exp18/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$HOME/venvs/exp18/bin/pip" install -q numpy scipy matplotlib pyarrow pillow
"$HOME/venvs/exp18/bin/python" -c "import torch; print('torch', torch.__version__)"
