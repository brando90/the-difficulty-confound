#!/usr/bin/env bash
# CPU end-to-end test of the GPU plans. Never a result.
#   stage 1  toy data (prep_data.py --toy; downloads only the Qwen2.5 tokenizer, for K)
#   stage 2  smoke-tier runs: a tiny random Qwen2 model, 8 steps, every evaluation (seed 0 by default; SEEDS=all for both)
#   stage 3  planted-effect test: fake runs with known answers -> each analyze.py -> verify.py (must print VERIFY PASS)
#
#   PY=~/.venvs/tdc/bin/python bash experiments/gpu/common/smoke_test.sh [workdir] [stages, default 123]
# Single-threaded and niced, so it can share a busy machine; stage 2 takes a few minutes per run on one core.
set -euo pipefail
G=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=${1:-${TMPDIR:-/tmp}/tdc_gpu_smoke}
STAGES=${2:-123}
PY=${PY:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-$W/hf}
RUN="nice -n 19 $PY"
mkdir -p "$W"
echo "smoke test in $W (stages $STAGES)"

if [[ $STAGES == *1* ]]; then
  [[ -f $W/data/v1/manifest.json ]] && echo "stage 1: toy data already built" || $RUN "$G/common/prep_data.py" --data-root "$W/data" --toy
fi

if [[ $STAGES == *2* ]]; then
  N=$($PY "$G/common/runs.py" count --tier smoke --plans g1,g2,g3 --seeds "${SEEDS:-0}")
  for ((i = 0; i < N; i++)); do
    $RUN "$G/common/train.py" --tier smoke --plans g1,g2,g3 --seeds "${SEEDS:-0}" --index "$i" \
      --data-root "$W/data" --out-root "$W/out" >"$W/smoke_$i.log" 2>&1 || { echo "stage 2: run $i FAILED, see $W/smoke_$i.log"; exit 1; }
    tail -1 "$W/smoke_$i.log"
  done
  $PY "$G/common/runs.py" status --tier smoke --plans g1,g2,g3 --seeds "${SEEDS:-0}" --out-root "$W/out"
fi

if [[ $STAGES == *3* ]]; then
  F=$W/fake
  rm -rf "$F" && mkdir -p "$F"
  $RUN "$G/common/fake_runs.py" --out-root "$F/out" --data-root "$F/data" --tier main
  $RUN "$G/g1_hard_to_easy_putnam_style_vs_gsm8k/analyze.py" --out-root "$F/out" --tier main --report-dir "$F/g1" >/dev/null
  $RUN "$G/g2_cross_domain_math_vs_poetry/analyze.py" --out-root "$F/out" --tier main --report-dir "$F/g2" >/dev/null
  $RUN "$G/g3_llm_complexity_vs_alignment_elasticity/analyze.py" --out-root "$F/out" --data-root "$F/data" --tier main --secondary --report-dir "$F/g3" >/dev/null
  ok=1
  for p in g1 g2 g3; do
    extra=()
    [[ $p == g3 ]] && extra=(--data-root "$F/data")
    if out=$($RUN "$G/common/verify.py" --plan $p --tier main --out-root "$F/out" ${extra[@]+"${extra[@]}"} --report-dir "$F/$p"); then :; else ok=0; fi
    echo "$p: $(tail -1 <<<"$out")"
  done
  ((ok)) || { echo "stage 3: VERIFY FAIL"; exit 1; }
  echo "stage 3: planted effects recovered; reports in $F/g1 $F/g2 $F/g3 (planted values: common/fake_runs.py)"
fi
echo "smoke test passed"
