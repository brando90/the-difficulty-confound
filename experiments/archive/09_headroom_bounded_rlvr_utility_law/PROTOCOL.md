# Protocol: Headroom-Bounded RLVR Utility Laws and Group-Size Interventions (`09_headroom_bounded_rlvr_utility_law`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/09_headroom_bounded_rlvr_utility_law/PROTOCOL.md>

**TLDR:** Experiment 08's additive RL utility law predicted more improvement than was possible at width 384. We cap the forecast at the remaining headroom, $`\widehat U = (1-\text{Acc}_0)(1-e^{-r_\theta})`$, and test on fresh seeds whether the cap fixes extrapolation and whether probe features still help inside it. We also vary the rollout group size $G$ for the same starting checkpoints, to test whether signal density $\rho(G)$ actually drives RL utility or just travels with other variables.

Author: Laolu Oguneye

Frozen on 10-18-2026 before any Experiment 09 training.

## 1. Scientific Motivation

1. **The additive law has no ceiling.** In [Experiment 08](../archive/08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md), probe features cut RL utility RMSE by **48.56%** at width 96 but made it worse at width 384 (pooled **-15.78%**, `NOT_ESTABLISHED`). At width 384 models started near ceiling with about 1.8 pp of headroom left, yet even `utility_base` forecast 18.6 pp of gain, and 127/192 forecasts exceeded $1-\text{Acc}_0$. Nothing in $\widehat U = b_0 + b_N\log N + b_C s(C) + \dots$ ties the forecast to how much room is left.
2. **Headroom-bounded law.** Write utility as remaining headroom times the fraction recovered:

   ```math
   \widehat U = (1-\text{Acc}_0)\big(1 - e^{-r_\theta}\big), \qquad r_\theta = s(C)\,\exp(\theta^\top\phi), \qquad s(C) = \frac{C}{C+300}.
   ```

   Forecasts always satisfy $0 \le \widehat U \lt 1-\text{Acc}_0$, start at 0 when $C=0$, and level off as training continues. This is the form proposed in [`COLLABORATION_ONBOARDING.md`](../ideas/COLLABORATION_ONBOARDING.md). Its known weakness is that it cannot predict a score drop: 29% of Experiment 08 test rows had $U \lt 0$, mostly early in RL. We report its error on those rows separately.
3. **Signal density was only observed, never controlled.** In Experiment 08, $\rho = 1 - p^G - (1-p)^G$ moved together with width, history and starting accuracy, so we can't tell whether it matters on its own. Changing $G$ for the *same* starting checkpoint changes $\rho$ while holding everything else fixed. We keep total rollouts per update at $B \cdot G = 4096$, so $G=16$ reproduces Experiment 08 exactly. Under that budget, larger groups give more informative groups per prompt but fewer prompts, so we make no prediction about which $G$ is best. We only ask whether knowing $\rho(G)$ forecasts held-out group sizes better than knowing $G$ alone.
4. **Exploratory disclosure.** Before writing this protocol we refit candidate laws on Experiment 08's published outcomes ([`exploratory/refit_exp08.py`](exploratory/refit_exp08.py)). On that already-seen data, bounded laws cut utility RMSE from 17.34 to about 5 pp with no impossible forecasts, and width-384 errors sat at the binomial noise floor (about 0.4 pp). Single-feature ablations were added after a first look. This shaped the noise-aware margin in Section 3 and our decision to drop a two-sided gain-minus-forgetting form. It is not evidence for this experiment; the fresh seeds are.

## 2. Experimental Design

- **Same task and data as Experiment 08:** byte-identical `addition.py`, network, optimizer, 180 answer-matched sums, and carry-marginal matched mixtures (50% ones carries, 50% tens carries in every mixture). Fresh evaluation split (`split_seed=20261029`): 2,880 test, 1,440 calibration, 1,440 probe items.
- **Seeds:** `9101, 9202, 9303, 9404, 9505, 9606, 9707, 9808`; analysis seed `29102026`.
- **Widths:** fit `32, 64, 128`; held-out `96, 384`. **Pretraining histories ($q_{\text{pre}}$):** fit `0.10, 0.50, 0.90`; held-out `0.20, 0.80`. 1,500 supervised updates each, giving 200 starting checkpoints.
- **Probe (before any RL):** starting bits $D_{m_0}$, sampled pass@1 $p$, and $\rho(G)$ for every group size, measured on the 1,440 probe items.
- **RLVR ($q_{\text{env}}=0.50$, 1,000 updates, checkpoints `100, 300, 1000`):** each starting checkpoint is trained once per group size, always from the same saved weights. 800 trajectories in total.

| $G$ | $B$ | Role |
|---|---|---|
| 4 | 1,024 | fit |
| 8 | 512 | held-out |
| 16 | 256 | fit; primary slice (same as Experiment 08) |
| 64 | 64 | held-out |

- **Laws** (fit per seed on calibration rows, scored on test rows; all forecasts frozen before test outcomes are read):
  - `additive_base`, `additive_signal_difficulty`: Experiment 08's utility laws, unchanged.
  - `bounded_base`: rate features are $\log N$ and the regime indicator.
  - `bounded_signal_difficulty`: adds $\log\rho(16)$, probe headroom $1-p$, and $\log D_{m_0}$.
  - `bounded_base_g`, `bounded_signal_g`: the same pair with $\log G$ added to both, fit on $G \in \{4, 16\}$ and scored on $G \in \{8, 64\}$.
  - `logit_base`, `logit_signal_difficulty` (comparison only, no gate): a signed bounded alternative that can predict drops.

  Fitting is least squares on $U$ with Experiment 08's ridge penalty ($\lambda = 0.01$). An impossible forecast is one above $1-\text{Acc}_0$, below $-\text{Acc}_0$, or nonfinite.

## 3. Prospective Pass/Fail Gates

Intervals are 95% paired bootstrap intervals over the 8 seed blocks (20,000 resamples). Because near-ceiling targets are dominated by sampling noise, an extended law counts as **non-inferior** on a held-out group if its RMSE is at most the baseline's RMSE plus $\sigma$, the binomial noise in observed $U$ for that group (about 0.4 pp at width 384 in Experiment 08).

1. **Gate 1 — Structural repair (`bounded_base` vs `additive_base`, $G=16$, 192 joint targets):** `SUPPORTED` iff relative RMSE reduction $\ge 10.0\%$, 95% lower bound $> 0\%$, both held-out widths improve, and 0 impossible forecasts.
2. **Gate 2 — Probe features inside the bound (`bounded_signal_difficulty` vs `bounded_base`, $G=16$, 192 joint targets):** `SUPPORTED` iff relative RMSE reduction $\ge 10.0\%$, 95% lower bound $> 0\%$, non-inferior at both held-out widths, and 0 impossible forecasts.
3. **Gate 3 — Signal density forecasts held-out group sizes (`bounded_signal_g` vs `bounded_base_g`, 864 targets) [secondary]:** `SUPPORTED` iff relative RMSE reduction $\ge 10.0\%$, 95% lower bound $> 0\%$, non-inferior at both $G=8$ and $G=64$, and 0 impossible forecasts.
4. **Gates 4–5 — Replications [secondary]:** Experiment 08's utility reversal and RLVR remediation/retention gates, rerun on the $G=16$ slice with Experiment 08's exact criteria.

Every gate is reported with equal prominence whichever way it falls, and no seed, trajectory or forecast is dropped. Runs on Sherlock, CPU only (one `sbatch` job, 16 CPUs, 3-hour limit). `verify.py` replays all 1,000 saved checkpoints before results are reported.
