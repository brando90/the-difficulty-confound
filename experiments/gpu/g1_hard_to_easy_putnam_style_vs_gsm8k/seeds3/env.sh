# g1 seeds3 environment on Sherlock: source it. Own wheel-only venv under $G1S3 (never shared with other experiments).
# Sherlock is glibc 2.17: only manylinux2014 wheels install, so torch is pinned to 2.6.0 (+cu124 from PyPI).
module load python/3.12.1
export G1S3=${G1S3:-$SCRATCH/tdc_g1s3}
export VENV=${VENV:-$G1S3/venv}
export REPO=$G1S3/the-difficulty-confound
export GPU_DIR=$REPO/experiments/gpu
export S3=$GPU_DIR/g1_hard_to_easy_putnam_style_vs_gsm8k/seeds3
export DATA=${DATA:-$SCRATCH/tdc_gpu/data}      # the pilot's data, read only; checked against the pilot's manifest hash
export OUT=${OUT:-$G1S3/runs}
export HF_HOME=${HF_HOME:-$G1S3/hf_cache}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
DATA_MANIFEST_SHA256=168634321ec11542e615975b5d570683709a57d87e707a7af82e4929103ed997  # pilot summary.json
if [ ! -f "$VENV/.ready" ]; then
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip &&
  "$VENV/bin/pip" install -q --only-binary=:all: torch==2.6.0 numpy==2.2.6 transformers==4.57.6 tokenizers==0.22.2 \
    "safetensors>=0.4.3" huggingface_hub==0.36.2 math-verify==0.9.0 &&
  "$VENV/bin/python" -c "import torch, numpy, transformers, math_verify; print('venv ok', torch.__version__, transformers.__version__)" &&
  touch "$VENV/.ready"
fi
PY=$VENV/bin/python
