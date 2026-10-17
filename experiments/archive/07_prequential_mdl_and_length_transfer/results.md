# Experiment 07 Results: Carry-Marginal Matched Heterogeneous Pretraining, No-Harm Remediation, and Prequential Online MDL

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/07_prequential_mdl_and_length_transfer/RESULTS_mdl_history.md>

**Frozen commit:** `b4044e8097ef3919178a34c63f2a66628902eb45` · **Verification:** `PASS` (`max_replay_bits_difference = 0.000e+00`, 200 trajectories, 200 final + 200 warm checkpoints replayed).

---

## 1. Executive Summary of Prospective Gates

| Prospective Gate | Comparison / Metric | Estimate [95% Seed-Block CI] | Gate Criterion | Status |
|---|---|---|---|---|
| **Gate 1: Conditional Loss Predictive Feature Value** | `stable_base` $\to$ `stable_difficulty` (joint held-out RMSE) | **36.22% [32.45, 40.51]** (`0.1470 -> 0.0938` bits) | $\ge 10.0\%$, lower CI $>0$, both widths improve, 0 invalid | **`SUPPORTED`** |
| **Gate 2a: Prequential Online MDL Reversal** | Online adaptation code-length gap (`pure_mdl_low` / `pure_mdl_high`) | **0.4423 [0.4043, 0.4824]** / **0.1773 [0.1525, 0.2016]** bits/item | Both $> 0$, lower CIs $>0$, all 5 widths $>0$ | **`SUPPORTED`** |
| **Gate 2b: Prequential Online MDL Prediction** | `stable_base` $\to$ `stable_difficulty` (joint prequential RMSE) | **56.93% [55.01, 58.67]** (`0.1823 -> 0.0785` bits) | $\ge 10.0\%$, lower CI $>0$ | **`SUPPORTED`** |
| **Gate 3: Pure Compositional Reversal (`step=0`)** | `(1,0)+(0,1)` vs `(1,1)` at $q_{\text{pre}}=.1$ / $.9$ | **46.06 [44.04, 48.26] pp** / **42.05 [39.95, 43.83] pp** | Both $\ge 10.0$ pp, lower CIs $>0$, all 5 widths $>0$ | **`SUPPORTED`** |
| **Gate 4: Compositional Remediation & Retention** | $q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$ (`step=1000`) | Recovery **104.29% [101.45, 107.53]**; control $\Delta$ **+5.13 [+4.13, +6.38] pp** | Recovery lo $\ge 50\%$, target $\ge 85\%$, control $\Delta \ge -5.0$ pp | **`NOT_ESTABLISHED`** |

---

## 2. Resolving the Shared-Warmup vs Heterogeneous-History Contrast Under Invariant Carry Marginals

In [Experiment 06](../06_compositional_carry_marginal_matched/RESULTS_compositional.md), holding exact answers and per-column carry frequencies ($50\%$ ones, $50\%$ tens; input-digit TV `0.0398`) fixed across all mixtures yielded only a **1.94% [-3.10, 6.50]** RMSE reduction from `stable_difficulty` because all models shared the same balanced warmup (`q=0.50`).

When checkpoints in Experiment 07 are pretrained across **heterogeneous carry-marginal matched histories** ($q_{\text{pre}} \in \{.1, .2, .5, .8, .9\}$) before balanced post-training ($q_{\text{post}}=0.50$):
- **`stable_base` joint held-out RMSE:** `0.1470 [0.1312, 0.1626]` bits/answer
- **`stable_oracle_history` joint held-out RMSE:** `0.1191 [0.0988, 0.1377]` bits/answer
- **`stable_difficulty` joint held-out RMSE:** `0.0938 [0.0803, 0.1062]` bits/answer
- **Relative RMSE reduction (`stable_base` $\to$ `stable_difficulty`):** **36.22% [32.45, 40.51]** (`SUPPORTED`), also outperforming `stable_oracle_history` by **21.26% [16.50, 25.17]**.

---

## 3. Prequential (Online) MDL Code Length Along the Adaptation Stream

Following [Voita & Titov (2020)](https://aclanthology.org/2020.emnlp-main.14/) and our complexity note ([`COMPLEXITY.md`](../01_fixed_width_addition_exposure/COMPLEXITY.md)), we measured the cumulative online prequential description length $\text{MDL}_{\text{preq}, s}(T)$ required to transmit the 1,000-step balanced adaptation stream ($256,000$ online examples) before each weight update:
- **After $q_{\text{pre}}=0.10$ (single-carry heavy):** Online MDL is **1.0981 [1.0202, 1.1758]** bits/item on $S_{\text{iso}}$ vs **1.5464 [1.4652, 1.6241]** bits/item on $S_{\text{comp}}$ (excess adaptation cost **0.4483 [0.4192, 0.4800]** bits/item on $S_{\text{comp}}$, and **0.4423 [0.4043, 0.4824]** bits/item on pure `(1,1)` vs `(1,0)+(0,1)`).
- **After $q_{\text{pre}}=0.50$ (matched balanced history):** Online MDL is **1.0594 [0.9460, 1.1787]** bits/item on $S_{\text{iso}}$ vs **1.1684 [1.0493, 1.2921]** bits/item on $S_{\text{comp}}$.
- **After $q_{\text{pre}}=0.90$ (double-carry heavy):** Online MDL reverses to **1.2831 [1.2053, 1.3537]** bits/item on $S_{\text{iso}}$ vs **1.0979 [1.0249, 1.1631]** bits/item on $S_{\text{comp}}$ (excess adaptation cost **0.1852 [0.1674, 0.2026]** bits/item on $S_{\text{iso}}$, and **0.1773 [0.1525, 0.2016]** bits/item on pure `(1,0)+(0,1)` vs `(1,1)`).
- **Predicting Online Prequential MDL on Held-Out Scales & Histories:** Adding probed starting difficulty (`stable_difficulty` vs `stable_base`) reduces joint held-out prequential MDL RMSE from **0.1823 [0.1629, 0.2016]** to **0.0785 [0.0689, 0.0888]** bits/item—a **56.93% [55.01, 58.67]** reduction (`SUPPORTED`).

---

## 4. Pure Compositional Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$)

- **Pre-intervention (`step=0`):** Single-carry control `(1,0)+(0,1)` accuracy is **79.11% [77.53, 80.76]** vs cascading two-carry target `(1,1)` accuracy **33.05% [32.05, 33.94]** (initial gap **46.06 [44.04, 48.26] pp**).
- **Post-remediation (`step=1000`):** Cascading two-carry `(1,1)` accuracy rises to **81.09% [79.51, 82.88]** (**+48.03 [+46.18, +50.19] pp**, recovering **104.29% [101.45, 107.53]** of the initial gap), while single-carry control `(1,0)+(0,1)` accuracy changes by **+5.13 [+4.13, +6.38] pp** to **84.24% [82.64, 85.83]** (`NOT_ESTABLISHED`).
