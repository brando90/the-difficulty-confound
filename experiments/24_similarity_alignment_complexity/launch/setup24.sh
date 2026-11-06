#!/usr/bin/env bash
# One-time Experiment 24 setup on SNAP skampere2: sparse clone (Experiments 13 and 24 only) at the given commit on DFS,
# then the autopilot in tmux session tdc_exp24. Usage, on skampere2: bash setup24.sh <commit sha>
set -euo pipefail
[[ $(hostname -s) == skampere2 ]] || { echo "run on skampere2"; exit 1; }
SHA=${1:?commit sha}
B=/dfs/scratch1/$USER/tdc_exp24
VENV=/lfs/skampere2/0/$USER/venvs/tdc_exp13
[[ -x $VENV/bin/python ]] || { echo "missing $VENV (Experiment 13's skampere2 environment)"; exit 1; }
mkdir -p "$B"
cd "$B"
if [[ ! -d the-difficulty-confound ]]; then
  git clone -q --depth 1 --filter=blob:none --no-checkout --branch exp24-similarity-alignment \
    https://github.com/brando90/the-difficulty-confound.git the-difficulty-confound
  git -C the-difficulty-confound sparse-checkout set experiments/13_ginc_hmm_complexity_vs_alignment \
    experiments/24_similarity_alignment_complexity
fi
git -C the-difficulty-confound fetch -q --depth 1 origin exp24-similarity-alignment
git -C the-difficulty-confound checkout -q --detach "$SHA"
[[ -z $(git -C the-difficulty-confound status --porcelain --untracked-files=no) ]]
tmux has-session -t tdc_exp24 2>/dev/null && { echo "tdc_exp24 already running"; exit 0; }
tmux new-session -d -s tdc_exp24 "bash $B/the-difficulty-confound/experiments/24_similarity_alignment_complexity/launch/autopilot24.sh"
echo "started tmux session tdc_exp24 at $(git -C the-difficulty-confound rev-parse --short HEAD)"
