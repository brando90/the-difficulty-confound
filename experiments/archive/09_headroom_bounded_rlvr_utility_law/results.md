# Experiment 09 Results: Headroom-Bounded RLVR Utility Laws and Group-Size Interventions

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/09_headroom_bounded_rlvr_utility_law/RESULTS_headroom_bounded.md>

**Frozen commit:** `d25e66919701464b9045dfca7b2c9e1126e2f55e` · **Verification:** `PASS` (`max_replay_bits_difference = 0.000e+00`, 200 starting checkpoints, 800 RLVR trajectories, 800 final + 200 warm checkpoints replayed, frozen forecasts reproduced, independent-optimizer gap `4.7e-12`).

---

## 1. Executive Summary of Prospective Gates

| Gate | Comparison | Estimate [95% seed-block CI] | Criterion | Status |
|---|---|---|---|---|
| **Gate 1: Structural repair** | `additive_base` → `bounded_base` ($G=16$, 192 joint targets) | **73.72% [70.43, 77.26]** (`17.62 -> 4.63` pp; 128 → 0 impossible forecasts) | $\ge 10\%$, lower CI $>0$, both widths improve, 0 impossible | **`SUPPORTED`** |
| **Gate 2: Probe features inside the bound** | `bounded_base` → `bounded_signal_difficulty` ($G=16$, 192 joint targets) | **8.10% [2.52, 14.39]** (`4.63 -> 4.26` pp) | $\ge 10\%$, lower CI $>0$, non-inferior at both widths, 0 impossible | **`NOT_ESTABLISHED`** |
| **Gate 3: Signal density across group sizes** | `bounded_base_g` → `bounded_signal_g` (864 targets, $G \in \{8, 64\}$) | **0.96% [0.44, 1.54]** (`9.81 -> 9.72` pp) | $\ge 10\%$, lower CI $>0$, non-inferior at both $G$, 0 impossible | **`NOT_ESTABLISHED`** |
| **Gate 4: RL utility reversal (replication)** | `u_low` / `u_high` at step 1000, $G=16$ | **29.86 [26.75, 32.59] pp** / **34.49 [31.68, 37.09] pp** | Both $\ge 10$ pp, lower CIs $>0$, all 5 widths $>0$ | **`SUPPORTED`** |
| **Gate 5: RLVR remediation and retention (replication)** | $q_{\text{pre}}=0.10$, step 1000, $G=16$ | Recovery **77.66% [72.94, 82.21]**; control $\Delta$ **+5.92 [+4.87, +7.01] pp** | Recovery lo $\ge 50\%$, control $\Delta \ge -5$ pp | **`SUPPORTED`** |

Intervals are 95% percentile intervals from 20,000 paired bootstrap resamples of the 8 seed blocks, conditional on the fixed test set; p-val=n/a. RMSEs are in percentage points of RL utility $U = \text{Acc}_T - \text{Acc}_0$.

---

## 2. Bounding by Headroom Repairs Extrapolation; Probe Features Add Less Than Hoped

| Law | Joint RMSE | Width 96 | Width 384 | Impossible forecasts |
|---|---|---|---|---|
| `additive_base` (Experiment 08) | 17.62 [15.91, 18.98] | 13.97 | 20.63 | 128 / 192 |
| `additive_signal_difficulty` (Experiment 08) | 21.10 [18.83, 23.15] | 7.27 | 28.95 | 99 / 192 |
| `bounded_base` | 4.63 [3.85, 5.32] | 6.53 | 0.41 | 0 |
| `bounded_signal_difficulty` | 4.26 [3.51, 4.91] | 6.00 | 0.48 | 0 |
| `logit_base` (comparison, no gate) | 4.09 [3.21, 4.88] | 5.77 | 0.38 | 0 |
| `logit_signal_difficulty` (comparison, no gate) | 4.38 [3.43, 5.24] | 6.19 | 0.37 | 0 |

- **The failure from Experiment 08 replicates on fresh seeds, and the bound removes it.** At width 384, `additive_base` again forecast far more gain than was possible (signed bias +18.48 pp; 94/96 forecasts above the remaining headroom). `bounded_base` misses by 0.41 pp, equal to the binomial noise in observed $U$ at that width ($\sigma = 0.41$ pp). It also improves at width 96 (13.97 → 6.53 pp).
- **Probe features help modestly.** Adding $\log\rho(16)$, probe headroom and $\log D_{m_0}$ to the bounded law lowers joint RMSE by 8.10%. The interval excludes zero, but the estimate is below the frozen 10% threshold. The gain comes from width 96 (6.53 → 6.00 pp, 8.21%); at width 384 the extended law is slightly worse (0.41 → 0.48 pp, -15.56%) but within the noise margin. The exploratory refit on Experiment 08's outcomes had suggested 16.53% for the same comparison; the fresh-seed estimate is about half that.
- **The signed logit comparator has the lowest joint RMSE (4.09 pp)**, and adding probe features to it makes it worse (-7.15% [-11.50, -3.22]). Once a law starts from $\text{Acc}_0$, the probe features carry little extra information in this setting. This is an observation, not a tested claim.
- **Score drops.** On the 59 joint targets with observed $U < 0$, `bounded_base` errs by 3.98 pp with bias +2.67 pp, as expected for a form that cannot forecast negative utility.

---

## 3. Group Size Changes Utility, but Not Through Signal Density Alone

At a fixed budget of 4,096 rollouts per update, larger groups raise probe signal density but lower realized utility (seed-mean $U$ at step 1000, all widths and histories):

| $G$ | Prompts per update | Probe $\rho(G)$ | Online mixed-reward fraction | Mean $U$ (pp) | Width 32 | Width 384 |
|---|---|---|---|---|---|---|
| 4 | 1,024 | 0.415 | 0.198 | 19.08 | 27.0 | 4.3 |
| 8 | 512 | 0.568 | 0.305 | 19.20 | 28.0 | 4.3 |
| 16 | 256 | 0.691 | 0.427 | 18.39 | 24.6 | 4.3 |
| 64 | 64 | 0.852 | 0.665 | 14.89 | 13.7 | 4.2 |

- **More rollouts per prompt meant fewer prompts and less improvement**, concentrated in the smallest network (width 32: 27.0 → 13.7 pp). Near ceiling (width 384), $G$ made no difference.
- **Knowing $\rho(G)$ adds almost nothing beyond knowing $G$** (Gate 3: 0.96%; `group_g8` 8.79 → 8.76 pp, `group_g64` 10.74 → 10.60 pp). Both laws over-predict held-out group sizes, mostly at $G=64$ (bias +6.10 / +5.57 pp, versus +1.68 / +1.89 pp at $G=8$), because the drop in utility at large $G$ falls outside the fitted range $G \in \{4, 16\}$.
- **Interpretation.** Under a fixed rollout budget, prompt diversity mattered more than per-prompt signal. Signal density as measured at the starting checkpoint does not, on its own, explain how group size changes RL utility here. On the 384 triple-held-out targets (width, history and $G$ all unseen) the signal law did better, 6.00% [2.26, 10.51], which is descriptive only.

---

## 4. Experiment 08 Findings Replicate on Fresh Seeds ($G=16$)

- **Utility reversal (Gate 4).** At $q_{\text{pre}}=0.10$, RLVR improves the scarce paired regime $S_{\text{comp}}$ by **29.86 pp** more than $S_{\text{iso}}$; at $q_{\text{pre}}=0.90$ the direction flips, and $S_{\text{iso}}$ gains **34.49 pp** more. Positive at all five widths. Experiment 08: 29.07 / 32.70 pp.
- **Remediation and retention (Gate 5).** From $q_{\text{pre}}=0.10$ checkpoints, cascading two-carry `(1,1)` accuracy rises from **33.07%** to **68.41%** (+35.34 pp, **77.66%** of the 45.51 pp initial gap), while single-carry control accuracy rises from **78.57%** to **84.50%** (+5.92 pp). Experiment 08: 78.23% recovery, +6.45 pp control.

---

## 5. Notes and Limitations

- **Exploratory influence.** The bounded form was proposed before any refit, but the noise-aware margin and the decision to drop a two-sided form were informed by a disclosed refit on Experiment 08's published outcomes ([`exploratory/`](exploratory/)). Gate 2's fresh-seed estimate (8.10%) is lower than that refit (16.53%).
- **Noise margin.** Gate 2 at width 384 and Gate 3 at both group sizes count as non-inferior only because of the frozen margin of one binomial noise standard deviation. Neither gate fails on that criterion; both fail on the 10% joint threshold.
- **Ridge penalty.** All laws reuse Experiment 08's ridge penalty ($\lambda=0.01$) on standardized coefficients, which is strong relative to squared error measured in utility units. It applies equally to every law, but different shrinkage could change the size of the Gate 2 effect.
- **Environment.** Trained on Sherlock (job `48325211`, CPU, 6 min 46 s, `elapsed_seconds = 406.28` in `runtime.json`) with Python 3.12.1, torch 2.5.1+cpu, numpy 1.26.4. Experiment 08 used Python 3.11.11 with the same torch and numpy versions.
- **Scope.** Small networks on fixed-width addition. These results do not by themselves validate or refute the `rl-4-code-verification` utility law on language models.

All 200 starting checkpoints, 800 RLVR trajectories and every forecast are retained.
