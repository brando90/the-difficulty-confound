# Experiment 11 Results: Calibrated Pretraining Horizon for Causal Autoregressive Transformers ($T_{\text{pre}}=300$)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/11_transformer_calibrated_pretraining_horizon/RESULTS_transformer_calibrated.md>

**Frozen commit:** `d765818eb8a37f5b40cf8cb6440263f35c24e754` · **Verification:** `PASS` (`max_replay_bits_difference = 0.000e+00`, 200 trajectories, 200 final + 200 warm checkpoints replayed).

---

## 1. Executive Summary of Prospective Gates

| Prospective Gate | Comparison / Metric | Estimate [95% Seed-Block CI] | Gate Criterion | Status |
|---|---|---|---|---|
| **Gate 1: Conditional Loss Predictive Feature Value** | `stable_base` $\to$ `stable_difficulty` (joint held-out RMSE) | **0.99% [-11.40, 5.66]** (`0.0674 -> 0.0667` bits) | $\ge 10.0\%$, lower CI $>0$, both widths improve, 0 invalid | **`NOT_ESTABLISHED`** |
| **Gate 2a: Prequential Online MDL Reversal** | Online adaptation code-length gap (`pure_mdl_low` / `pure_mdl_high`) | **0.0666 [0.0606, 0.0715]** / **-0.0006 [-0.0075, 0.0056]** bits/item | Both $> 0$, lower CIs $>0$, all 5 widths $>0$ | **`NOT_ESTABLISHED`** |
| **Gate 2b: Prequential Online MDL Prediction** | `stable_base` $\to$ `stable_difficulty` (joint prequential RMSE) | **18.84% [5.88, 34.16]** (`0.1626 -> 0.1320` bits) | $\ge 10.0\%$, lower CI $>0$ | **`SUPPORTED`** |
| **Gate 3: Pure Compositional Reversal (`step=0`)** | `(1,0)+(0,1)` vs `(1,1)` at $q_{\text{pre}}=.1$ / $.9$ | **15.67 [12.13, 19.10] pp** / **15.70 [12.89, 18.17] pp** | Both $\ge 10.0$ pp, lower CIs $>0$, all 5 widths $>0$ | **`SUPPORTED`** |
| **Gate 4: Compositional Remediation & Retention** | $q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$ (`step=1000`) | Recovery **201.55% [181.21, 234.28]**; control $\Delta$ **+15.94 [+14.18, +17.80] pp** | Recovery lo $\ge 50\%$, target $\ge 85\%$, control $\Delta \ge -5.0$ pp | **`NOT_ESTABLISHED`** |

---

## 2. Testing the Core Difficulty-Confound Thesis on Autoregressive Causal Transformers

All prior studies (Experiments 01–09) evaluated feedforward MLPs predicting all digits in parallel. Experiment 10 replicates the full Experiment 07 design on a **causal autoregressive transformer** (`AdditionTransformer`, 2 layers, 4 attention heads, causal self-attention, generating digits token-by-token from most to least significant).

When transformer checkpoints are pretrained across **heterogeneous carry-marginal matched histories** ($q_{\text{pre}} \in \{.1, .2, .5, .8, .9\}$) before balanced post-training ($q_{\text{post}}=0.50$):
- **`stable_base` joint held-out RMSE:** `0.0674 [0.0274, 0.1015]` bits/answer
- **`stable_oracle_history` joint held-out RMSE:** `0.0673 [0.0271, 0.1014]` bits/answer
- **`stable_difficulty` joint held-out RMSE:** `0.0667 [0.0272, 0.0999]` bits/answer
- **Relative RMSE reduction (`stable_base` $\to$ `stable_difficulty`):** **0.99% [-11.40, 5.66]** (`NOT_ESTABLISHED`), also comparing against `stable_oracle_history` (**0.89% [-12.19, 5.04]**).

---

## 3. Prequential (Online) MDL Code Length Along the Adaptation Stream

Measuring cumulative online prequential description length $\text{MDL}_{\text{preq}, s}(T)$ along the 1,000-step balanced adaptation stream for autoregressive transformers:
- **After $q_{\text{pre}}=0.10$ (single-carry heavy):** Online MDL is **0.2251 [0.1857, 0.2715]** bits/item on $S_{\text{iso}}$ vs **0.2957 [0.2619, 0.3358]** bits/item on $S_{\text{comp}}$ (excess adaptation cost **0.0706 [0.0634, 0.0772]** bits/item on $S_{\text{comp}}$, and **0.0666 [0.0606, 0.0715]** bits/item on pure `(1,1)` vs `(1,0)+(0,1)`).
- **After $q_{\text{pre}}=0.50$ (matched balanced history):** Online MDL is **0.3110 [0.2525, 0.3819]** bits/item on $S_{\text{iso}}$ vs **0.3377 [0.2763, 0.4090]** bits/item on $S_{\text{comp}}$.
- **After $q_{\text{pre}}=0.90$ (double-carry heavy):** Online MDL reverses to **0.2973 [0.2343, 0.3656]** bits/item on $S_{\text{iso}}$ vs **0.2991 [0.2341, 0.3701]** bits/item on $S_{\text{comp}}$ (excess adaptation cost **-0.0019 [-0.0083, 0.0038]** bits/item on $S_{\text{iso}}$, and **-0.0006 [-0.0075, 0.0056]** bits/item on pure `(1,0)+(0,1)` vs `(1,1)`).
- **Predicting Online Prequential MDL on Held-Out Scales & Histories:** Adding probed starting difficulty (`stable_difficulty` vs `stable_base`) reduces joint held-out prequential MDL RMSE from **0.1626 [0.1380, 0.1922]** to **0.1320 [0.1203, 0.1423]** bits/item—a **18.84% [5.88, 34.16]** reduction (`SUPPORTED`).

---

## 4. Pure Compositional Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$)

- **Pre-intervention (`step=0`):** Single-carry control `(1,0)+(0,1)` accuracy is **83.99% [82.12, 85.77]** vs cascading two-carry target `(1,1)` accuracy **68.32% [64.27, 72.35]** (initial gap **15.67 [12.13, 19.10] pp**).
- **Post-remediation (`step=1000`):** Cascading two-carry `(1,1)` accuracy rises to **99.90% [99.86, 99.94]** (**+31.58 [+27.57, +35.62] pp**, recovering **201.55% [181.21, 234.28]** of the initial gap), while single-carry control `(1,0)+(0,1)` accuracy changes by **+15.94 [+14.18, +17.80] pp** to **99.93% [99.90, 99.96]** (`NOT_ESTABLISHED`).
