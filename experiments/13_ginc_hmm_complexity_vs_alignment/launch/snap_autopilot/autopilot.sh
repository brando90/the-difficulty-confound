#!/usr/bin/env bash
# Experiment 13 (13-GPU) autopilot on skampere1. Runs unattended in tmux session tdc_exp13_auto:
#   1. waits for the 9 pilot jobs (started by snap.sh MODE=pilot at the GPU code commit);
#   2. applies the frozen pilot rule (run.py --pilot-select) and two sanity guards (all finite; chosen excess < 1 bit);
#   3. makes the "Experiment 13 GPU pilot disclosure" commit in the DFS clone (config gpu.lr_selected + a PROTOCOL
#      Section 7 paragraph only; not pushed; fetched to the laptop later, same SHA);
#   4. runs the 360 shards with one worker per shared GPU (largest models first, claims by mkdir, each shard verified on
#      its GPU right after training; a worker waits while its GPU has < MIN_FREE_MB free);
#   5. when all 360 are DONE and VERIFIED: merge, analyze (stages 1 and 2) and merged verify, on CPU.
# Any failure writes $STATE/HALT with the reason and stops before the next stage. No automatic reruns of failed shards.
set -uo pipefail
U=laoluog1
B=/dfs/scratch0/$U/tdc_exp13
REPO=$B/the-difficulty-confound
EXP=experiments/13_ginc_hmm_complexity_vs_alignment
CODE=$REPO/$EXP/expt_v1
VENV=/lfs/skampere1/0/laoluog1/venvs/tdc_exp13
PY=$VENV/bin/python
PILOT=$B/runs/gpu_v1/pilot
OUT=${OUT:-/dfs/scratch1/$U/tdc_exp13/runs/gpu_v1}
STATE=$B/state
LOG=$B/runs/gpu_v1/logs/autopilot.log
CODE_SHA=2c625111
MIN_FREE_MB=${MIN_FREE_MB:-24000}
MAX_SHARD_GPUS=${MAX_SHARD_GPUS:-4}
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv TORCH_HOME=$B/cache/torch \
  MPLCONFIGDIR=$B/cache/mpl PIP_CACHE_DIR=/lfs/skampere1/0/laoluog1/pip_cache
mkdir -p "$STATE" "$OUT/logs" "$OUT/shards" "$OUT/claims" "$B/cache"
say() { echo "$(date '+%m-%d-%Y %H:%M:%S') $*" | tee -a "$LOG"; }
halt() { say "HALT: $*"; echo "$(date '+%m-%d-%Y %H:%M:%S') $*" >"$STATE/HALT"; exit 1; }
[[ ! -f $STATE/HALT ]] || { say "HALT file present ($(cat "$STATE/HALT")); remove it to resume"; exit 1; }

# ---- 1. pilot ------------------------------------------------------------------------------------------------------
if [[ ! -f $STATE/DISCLOSURE_SHA ]]; then
  say "waiting for the 9 pilot jobs in $PILOT"
  while true; do
    n=$(ls "$PILOT"/*/DONE 2>/dev/null | wc -l)
    ((n == 9)) && break
    if [[ -f $B/runs/gpu_v1/logs/status_skampere1.tsv ]] &&
      awk -F'\t' '$2 == "pilot" && $5 != 0 { bad = 1 } END { exit !bad }' "$B/runs/gpu_v1/logs/status_skampere1.tsv"; then
      halt "a pilot job exited nonzero (see $B/runs/gpu_v1/logs/status_skampere1.tsv)"
    fi
    if ! tmux has-session -t tdc_exp13 2>/dev/null && ! tmux has-session -t tdc_exp13_setup 2>/dev/null && ((n < 9)); then
      halt "pilot workers are gone with $n of 9 DONE"
    fi
    sleep 120
  done
  say "pilot: 9 of 9 DONE"
  cd "$REPO" || halt "no repo"
  [[ $(git rev-parse --short=8 HEAD) == "$CODE_SHA" ]] || halt "repo HEAD is not $CODE_SHA"
  "$PY" "$CODE/run.py" --arm gpu --pilot-select "$PILOT" >>"$LOG" 2>&1 || halt "pilot-select failed"
  # ---- 2-3. guards and the disclosure commit ----------------------------------------------------------------------
  "$PY" - "$PILOT" "$REPO/$EXP" >>"$LOG" 2>&1 <<'PYEOF' || halt "pilot guards failed or disclosure edit failed (see log)"
import json, math, sys, time
from pathlib import Path
pilot, exp = Path(sys.argv[1]), Path(sys.argv[2])
sel = json.loads((pilot / 'pilot_selection.json').read_text())
res = sel['results']
bad = [r for r in res if not (r['finite'] and math.isfinite(r['excess_bits']))]
if bad:
    sys.exit(f'nonfinite pilot runs: {bad}')
chosen = sel['lr_selected']
for size, m in chosen.items():
    r = next(x for x in res if x['size'] == size and x['lr_multiplier'] == m)
    if r['excess_bits'] >= 1.0:
        sys.exit(f'guard: chosen {size} x{m} has excess {r["excess_bits"]:.3f} >= 1 bit at 2^26')
first = {}
for j in range(9):
    rec = json.loads((pilot / f'{j:05d}' / 'run.json').read_text())
    row = next(x for x in rec['rows'] if x['slice'] == 'all' and x['T'] == min(rec['T_checkpoints']))
    first[(rec['size'], rec['job']['lr_multiplier'])] = row['excess_bits']
gpus = sorted({r['gpu'] for r in res})
dates = sorted(json.loads((pilot / f'{j:05d}' / 'run.json').read_text())['started'][:10] for j in range(9))
cfg = exp / 'expt_v1' / 'config.json'
s = cfg.read_text()
assert s.count('"lr_selected": null') == 1
s = s.replace('"lr_selected": null', '"lr_selected": ' + json.dumps({k: chosen[k] for k in ('S', 'M', 'L')}))
cfg.write_text(s)
fmt = lambda size: ', '.join(f'{m}x {next(r["excess_bits"] for r in res if r["size"] == size and r["lr_multiplier"] == m):.4f}'
                             for m in (0.5, 1, 2))
lr = {r['size']: r['lr'] for r in res for k, m in chosen.items() if r['size'] == k and r['lr_multiplier'] == m}
par = (f'\n**GPU pilot disclosure (13-GPU, run {dates[0]} to {dates[-1]} at commit 2c625111 on SNAP skampere1, '
       f'{", ".join(gpus)} shared with other users).** Seed `13919`, the n = 32 pilot concept (`hmm_n032_p1`), '
       f'α = 0.4, T to $2^{{26}}$, learning rate 0.5×, 1× and 2× the Section 2.6 value per size (9 runs, none '
       f'nonfinite). Excess ε over all positions at $2^{{26}}$ (bits/token): S {fmt("S")}; M {fmt("M")}; L {fmt("L")}. '
       f'Rule of Section 2.8 (lowest ε at $2^{{26}}$, ties within 0.002 bits to 1×): **S {chosen["S"]}× '
       f'({lr["S"]:.1e}), M {chosen["M"]}× ({lr["M"]:.1e}), L {chosen["L"]}× ({lr["L"]:.1e})** (`gpu.lr_selected`). '
       f'Reported, not gated: ε at the first checkpoint ($2^{{22}}$) at 1×: S {first[("S", 1)]:.3f}, '
       f'M {first[("M", 1)]:.3f}, L {first[("L", 1)]:.3f} bits/token. Seed `13919` is never used again. '
       f'This commit changes only `gpu.lr_selected` and adds this paragraph; every shard runs at it.\n')
p = exp / 'PROTOCOL.md'
t = p.read_text()
anchor = '\n## 8. Artifacts and verification'
assert t.count(anchor) == 1
p.write_text(t.replace(anchor, par + anchor))
print('disclosure edits written:', json.dumps(chosen))
PYEOF
  git add "$EXP/expt_v1/config.json" "$EXP/PROTOCOL.md" || halt "git add failed"
  [[ $(git diff --cached --name-only | wc -l) == 2 ]] || halt "disclosure commit would touch other files"
  git -c user.name="Laolu Oguneye" -c user.email=laoluog@outlook.com commit -q -m "Experiment 13 GPU pilot disclosure" \
    -m "GPU pilot (seed 13919, n = 32 pilot concept, alpha 0.4, T to 2^26) on skampere1; frozen rule of PROTOCOL Section 2.8. Changes only config.json gpu.lr_selected and adds the GPU pilot disclosure paragraph to PROTOCOL Section 7. Every 13-GPU shard runs at this commit." ||
    halt "disclosure commit failed"
  git rev-parse HEAD >"$STATE/DISCLOSURE_SHA"
  say "disclosure commit $(cat "$STATE/DISCLOSURE_SHA"): $(grep -o '"lr_selected": {[^}]*}' "$CODE/config.json")"
fi
SHA=$(cat "$STATE/DISCLOSURE_SHA")
cd "$REPO" || halt "no repo"
[[ $(git rev-parse HEAD) == "$SHA" ]] || halt "repo HEAD is not the disclosure commit $SHA"
[[ -z $(git status --porcelain --untracked-files=no) ]] || halt "tracked files modified in $REPO"
ln -sfn "$PILOT" "$OUT/pilot"

# ---- 4. shards -----------------------------------------------------------------------------------------------------
if ! tmux has-session -t tdc_exp13_shards 2>/dev/null && [[ ! -f $STATE/SHARDS_LAUNCHED ]]; then
  tmux kill-session -t tdc_exp13 2>/dev/null  # the pilot workers are finished (9 DONE); their windows stay open otherwise
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' -v m="$MIN_FREE_MB" '$2 >= m { print $3, $1 }' | sort -n | head -n "$MAX_SHARD_GPUS" | awk '{ print $2 }')
  ((${#GPUS[@]} > 0)) || halt "no GPU with >= $MIN_FREE_MB MB free for the shards"
  say "shards: workers on GPUs ${GPUS[*]} (>= $MIN_FREE_MB MB free, least busy first), output $OUT"
  W=$B/shard_worker.sh
  for g in "${GPUS[@]}"; do
    if ! tmux has-session -t tdc_exp13_shards 2>/dev/null; then
      tmux new-session -d -s tdc_exp13_shards -n "gpu$g" "bash $W $g $OUT 2>&1 | tee -a $OUT/logs/worker_gpu$g.log"
      tmux set-option -t tdc_exp13_shards remain-on-exit on >/dev/null
    else
      tmux new-window -t tdc_exp13_shards -n "gpu$g" "bash $W $g $OUT 2>&1 | tee -a $OUT/logs/worker_gpu$g.log"
    fi
  done
  touch "$STATE/SHARDS_LAUNCHED"
fi
say "waiting for the shard workers"
while true; do
  live=$(tmux list-panes -t tdc_exp13_shards -F '#{pane_dead}' 2>/dev/null | grep -c '^0$')
  ((live > 0)) || break
  sleep 300
done
nd=$(ls "$OUT"/shards/*/DONE 2>/dev/null | wc -l)
nv=$(ls "$OUT"/shards/*/VERIFIED 2>/dev/null | wc -l)
say "shard workers finished: $nd DONE, $nv VERIFIED"
((nd == 360 && nv == 360)) || halt "$nd DONE and $nv VERIFIED of 360: failed shards need a manual rerun (claims kept)"

# ---- 5. merge, analysis, merged verify (CPU) -----------------------------------------------------------------------
RES=$CODE/results/gpu
if [[ ! -f $RES/manifest.json ]]; then
  "$PY" "$CODE/run.py" --arm gpu --merge "$OUT" >>"$LOG" 2>&1 || halt "merge failed"
fi
cd "$CODE" || halt "no code dir"
for st in 1 2; do
  OMP_NUM_THREADS=4 "$PY" -u analyze.py --arm gpu --stage $st --output "$RES" >>"$LOG" 2>&1 || halt "analyze stage $st failed"
done
OMP_NUM_THREADS=1 "$PY" -u verify.py --arm gpu --output "$RES" --workers 8 >>"$LOG" 2>&1 || halt "merged verify failed"
say "ALL_DONE: results in $RES; gates: $(tr -d '\n ' <"$RES/gates.json" | grep -o '"gates":{[^}]*}[^}]*}' | head -c 400)"
touch "$STATE/ALL_DONE"
