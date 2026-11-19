#!/usr/bin/env bash
# One-time Experiment 25 setup on SNAP skampere2: sparse clone (Experiments 13, 24 and 25) at the freeze commit on DFS,
# then the autopilot in tmux session tdc_exp25. Usage, on skampere2: bash setup25.sh <freeze commit sha>
set -euo pipefail
[[ $(hostname -s) == skampere2 ]] || { echo "run on skampere2"; exit 1; }
SHA=${1:?freeze commit sha}
B=/dfs/scratch1/$USER/tdc_exp25
VENV=/lfs/skampere2/0/$USER/venvs/tdc_exp13
BRANCH=exp25-alpha-elasticity-diagnosis
[[ -x $VENV/bin/python ]] || { echo "missing $VENV (Experiment 13's skampere2 environment)"; exit 1; }
mkdir -p "$B/state"
cd "$B"
if [[ ! -d the-difficulty-confound ]]; then
  git clone -q --depth 1 --filter=blob:none --no-checkout --branch "$BRANCH" \
    https://github.com/brando90/the-difficulty-confound.git the-difficulty-confound
  git -C the-difficulty-confound sparse-checkout set experiments/13_ginc_hmm_complexity_vs_alignment \
    experiments/24_similarity_alignment_complexity experiments/25_alpha_elasticity_diagnosis
fi
git -C the-difficulty-confound fetch -q --depth 1 origin "$BRANCH"
git -C the-difficulty-confound checkout -q --detach "$SHA"
[[ -z $(git -C the-difficulty-confound status --porcelain --untracked-files=no) ]]
[[ -f state/FREEZE_SHA ]] || git -C the-difficulty-confound rev-parse HEAD >state/FREEZE_SHA
[[ $(cat state/FREEZE_SHA) == $(git -C the-difficulty-confound rev-parse HEAD) ]] || { echo "state/FREEZE_SHA differs from HEAD"; exit 1; }
tmux has-session -t tdc_exp25 2>/dev/null && { echo "tdc_exp25 already running"; exit 0; }
tmux new-session -d -s tdc_exp25 "bash $B/the-difficulty-confound/experiments/25_alpha_elasticity_diagnosis/launch/autopilot25.sh"
echo "started tmux session tdc_exp25 at $(git -C the-difficulty-confound rev-parse --short HEAD)"
