# Experiment 25: why does diluting the target with unrelated data cost only half as much as it should?

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/25_alpha_elasticity_diagnosis/README.md>

In Experiment 24, even with distractors that shared no rules and no tokens with the target, a 20 times smaller target share needed only about 4.8 times more training ($e_\alpha = 0.52$, not 1). Every $E = e_\alpha / e_K$ in the series rests on this slope. This experiment tests four explanations on Experiment 24's own n = 8 and n = 32 targets and disjoint pools:

1. the single learning rate, tuned at α = 0.2, is too high for target-only data (each cell gets its own rate from a 5-point grid);
2. a batch of 128 target documents is more than a step needs (a masked arm, where distractor documents give no gradient);
3. disjoint HMMs still teach how to track a hidden state (the masked arm, plus an arm whose distractors keep only their unigram statistics);
4. part of the bill is paid by any tokens (does the slope steepen below α = 0.2?).

Grid: 2 sizes (S, M) × 6 targets × 16 cells (α = 1, plus 3 arms × α 0.02–0.5) × 5 learning rates × 3 seeds = 2,880 runs to 2^28 tokens, on SNAP skampere2. Gates G1–G5 (PROTOCOL Section 4.4).

Status: **COMPLETE 11-19-2026.** 2,880 of 2,880 runs verified; G2, G3, G4, G5 SUPPORTED, G1 NOT_ESTABLISHED (Laolu predicted all five). Dilution is cheap mainly because most of a run's bill is a fixed number of optimizer steps: even when the non-target documents give no gradient, e_α = 0.44 [0.39, 0.46]. Real unrelated languages lower it a further 0.07 (word soup does not), the slope climbs from 0.10 above α = 0.2 to 0.54 below it, and per-cell learning rates change it by −0.02 [−0.05, 0.02]. Results: [RESULTS_alpha_elasticity.md](RESULTS_alpha_elasticity.md).

Design: [PROTOCOL.md](PROTOCOL.md). Results: [RESULTS_alpha_elasticity.md](RESULTS_alpha_elasticity.md), figure `alpha_elasticity_results.png`, checksums `ARTIFACT_SHA256.json`. Run log: [CKPT_exp25.md](CKPT_exp25.md). Code: [`expt_v1/`](expt_v1/) (reuses Experiment 24's concepts and Tobit code and Experiment 13-GPU's model, stream and evaluation, unchanged). Launch: [`launch/`](launch/).

Reproduce: `python -m pytest -q expt_v1`; `python expt_v1/run25.py --check-only`; on skampere2, `bash launch/setup25.sh <freeze sha>` (starts `launch/autopilot25.sh` in tmux session `tdc_exp25`; after the merge it waits for `state/PREDICTIONS_SHA`); afterwards `analyze25.py` and `verify25.py` on the merged `expt_v1/results/`.

**TLDR-end:** [tdc: Expt 25 readme] Why does diluting a target with unrelated data cost half what it should (Experiment 24's $e_\alpha \approx 0.5$)? Mostly because a run needs a fixed number of optimizer steps whatever its batch holds: with distractors that give no gradient $e_\alpha$ is still 0.44, real unrelated languages lower it a little more, word soup does not, the slope steepens below a 20% share, and per-cell learning rates do not change it. G2–G5 SUPPORTED, G1 NOT_ESTABLISHED; 2,880 runs verified.
