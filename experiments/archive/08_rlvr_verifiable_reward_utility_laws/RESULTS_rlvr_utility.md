# Experiment 08 Results: Reinforcement Learning with Verifiable Exact-Sum Rewards (RLVR) and Utility Scaling Laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md>

**Frozen commit:** `95b9c80a5e871b4d0d512c92fb3a788d4f7fae0f` · **Verification:** `PASS` (`max_replay_bits_difference = 0.000e+00`, 200 RLVR trajectories, 200 final + 200 warm checkpoints replayed).

---

## 1. Executive Summary of Prospective Gates

| Prospective Gate | Comparison / Metric | Estimate [95% Seed-Block CI] | Gate Criterion | Status |
|---|---|---|---|---|
| **Gate 1: Post-RLVR Loss Predictive Feature Value** | `stable_base` $\to$ `stable_difficulty` (joint held-out RMSE) | **39.87% [36.20, 50.17]** (`0.7809 -> 0.4696` bits) | $\ge 10.0\%$, lower CI $>0$, both widths improve, 0 invalid | **`SUPPORTED`** |
| **Gate 2: Additive RL Utility Law (Pooled Joint)** | `utility_base` $\to$ `utility_signal_difficulty` (joint $U$ RMSE) | **-15.78% [-35.91, 3.15]** (`17.34 -> 20.08` pp; **+48.56% at w=96**, **-38.91% at w=384**) | $\ge 10.0\%$, lower CI $>0$, both widths improve | **`NOT_ESTABLISHED`** |
| **Gate 3: RL Utility Reversal Across Histories** | $U = \text{Acc}_{1000} - \text{Acc}_0$ contrast (`u_low` / `u_high`) | **29.07 [26.32, 32.33] pp** / **32.70 [30.18, 35.20] pp** | Both $\ge 10.0$ pp, lower CIs $>0$, all 5 widths $>0$ | **`SUPPORTED`** |
| **Gate 4: Pure 0/1 RLVR Remediation & Retention** | $q_{\text{pre}}=0.10 \to \text{RLVR } (q_{\text{env}}=0.50, G=16)$ | Recovery **78.23% [72.58, 84.23]**; control $\Delta$ **+6.45 [+4.69, +7.91] pp** | Recovery lo $\ge 50\%$, control $\Delta \ge -5.0$ pp | **`SUPPORTED`** |

---

## 2. Bridging to `rl-4-code-verification`: Why Additive RL Utility Laws Succeed at Interpolated Scales (`+48.56%`) but Overshoot Ceiling Headroom at Extrapolated Scales

In [`experiments/ideas/predictability_bridge/README.md`](../../ideas/predictability_bridge/README.md), we hypothesized that training history governs starting-policy success probabilities $p_i(m_0)$, which determine usable RL signal density $\rho_i(G) = 1 - p_i^G - (1-p_i)^G$, coverage $H_i(G) = 1 - (1-p_i)^G$, and starting difficulty $D_{m_0}$, thereby predicting downstream RL improvement utility $U = \text{Acc}_T - \text{Acc}_0$.

Evaluating the additive linear utility form from `rl-4-code-verification` across the 192 joint held-out targets reveals a sharp structural contrast between **scale interpolation (width 96)** and **near-ceiling scale extrapolation (width 384)**:

1. **At Interpolated Held-Out Scale (Width 96, 96 targets):**
   - `utility_base` RMSE: **14.59 [13.08, 16.07] pp**
   - `utility_oracle_history` RMSE: **8.53 [7.03, 9.98] pp** (**41.54% [36.38, 47.62]** reduction)
   - `utility_signal_coverage` (`+ log rho, headroom`) RMSE: **9.30 [7.62, 10.93] pp** (**36.26% [31.09, 42.41]** reduction)
   - `utility_signal_difficulty` (`+ log rho, headroom, log D_m0`) RMSE: **7.50 [5.70, 9.21] pp** — a **48.56% [41.65, 57.30] RMSE reduction** vs `utility_base`, and **12.01% [5.64, 22.71]** lower RMSE than `utility_oracle_history`!
2. **At Extrapolated Near-Ceiling Scale (Width 384, 96 targets) — Headroom Saturation Diagnosis:**
   - At width 384, pre-RLVR accuracy $\text{Acc}_0$ is already near ceiling ($78\%\text{–}99.8\%$), so true remaining headroom $1 - \text{Acc}_0$ is small ($0.2\text{–}22$ pp). Because the additive linear law $\widehat{U} = b_0 + b_N \log N + b_C s(C) + s(C)(\dots)$ does not multiply by remaining headroom $(1 - \text{Acc}_0)$, its non-negative budget slope $s(C)$ and probe terms fitted on lower-accuracy widths (`32, 64, 128`) over-predict $\widehat{U}$ at width 384 (`19.71 -> 27.38` pp, **-38.91% [-67.54, -12.56]**), causing pooled joint RMSE to be **20.08 [17.27, 22.52] pp** vs **17.34 [15.61, 18.85] pp** (**-15.78% [-35.91, 3.15]**, `NOT_ESTABLISHED`).
   - By contrast, our multiplicative constrained monotone ridge law on post-RLVR log loss (`stable_difficulty` vs `stable_base`) naturally respects the zero-loss floor and achieves a **39.87% [36.20, 50.17]** joint held-out RMSE reduction (`0.7809 -> 0.4696` bits/answer, **`SUPPORTED`**), improving at both held-out width 96 and held-out width 384!

---

## 3. Pure 0/1 Verifiable-Reward RL Remediation & Natural No-Harm Retention

During RLVR post-training, the model receives **only** a binary $0/1$ indicator of whether its 4 sampled digits equal the exact integer sum $a+b$, using leave-one-out group-relative advantage ($G=16$). Because already-mastered items have $\bar{R}_b = 1 \implies \hat{A}_{b,g} = 0$, group-relative RL naturally avoids gradient interference on mastered items while focusing updates on unmastered items with $\rho_i(G) > 0$:
- **Pre-RLVR (`step=0`, $q_{\text{pre}}=0.10$):** Single-carry control `(1,0)+(0,1)` accuracy is **77.81% [75.57, 80.79]** vs cascading two-carry target `(1,1)` accuracy **32.44% [29.45, 35.61]** (initial gap **45.37 [43.88, 46.91] pp**).
- **Post-RLVR (`step=1000`, $q_{\text{env}}=0.50$):** Cascading two-carry `(1,1)` accuracy rises to **67.93% [63.28, 72.89]** (**+35.49 [+33.27, +37.68] pp**, recovering **78.23% [72.58, 84.23]** of the initial gap), while single-carry control `(1,0)+(0,1)` accuracy changes by **+6.45 [+4.69, +7.91] pp** to **84.26% [82.88, 85.72]** (`SUPPORTED`).
- **RL Utility Reversal:** At $q_{\text{pre}}=0.10$, RLVR improves $S_{\text{comp}}$ by **+35.52 [+33.76, +37.34] pp** vs **+6.45 [+4.69, +7.91] pp** on $S_{\text{iso}}$ (`u_low = ` **29.07 [26.32, 32.33] pp**). At $q_{\text{pre}}=0.90$, the RL utility reverses: $S_{\text{iso}}$ improves by **+39.16 [+36.85, +41.83] pp** vs **+6.46 [+5.50, +7.36] pp** on $S_{\text{comp}}$ (`u_high = ` **32.70 [30.18, 35.20] pp**, `SUPPORTED`).
