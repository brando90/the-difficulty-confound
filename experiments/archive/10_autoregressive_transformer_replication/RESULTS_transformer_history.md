# Experiment 10 Results: Autoregressive Transformer Architecture Generalization (Heterogeneous Histories, Prequential Online MDL, and Pure Compositional Remediation)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/10_autoregressive_transformer_replication/RESULTS_transformer_history.md>

**Frozen commit:** `51c7f1c485cdcbe950a73de791e787190981529e` · **Verification:** `PASS` (`max_replay_bits_difference = 0.000e+00`, 200 trajectories, 200 final + 200 warm checkpoints replayed).

---

## 1. Executive Summary of Prospective Gates

| Prospective Gate | Comparison / Metric | Estimate [95% Seed-Block CI] | Gate Criterion | Status |
|---|---|---|---|---|
| **Gate 1: Conditional Loss Predictive Feature Value** | `stable_base` $\to$ `stable_difficulty` (joint held-out RMSE) | **0.05% [0.03, 2.84]** (`0.2242 -> 0.2241` bits) | $\ge 10.0\%$, lower CI $>0$, both widths improve, 0 invalid | **`NOT_ESTABLISHED`** |
| **Gate 2a: Prequential Online MDL Reversal** | Online adaptation code-length gap (`pure_mdl_low` / `pure_mdl_high`) | **0.0234 [0.0175, 0.0298]** / **0.0021 [0.0005, 0.0035]** bits/item | Both $> 0$, lower CIs $>0$, all 5 widths $>0$ | **`NOT_ESTABLISHED`** |
| **Gate 2b: Prequential Online MDL Prediction** | `stable_base` $\to$ `stable_difficulty` (joint prequential RMSE) | **3.77% [1.21, 6.93]** (`0.1200 -> 0.1155` bits) | $\ge 10.0\%$, lower CI $>0$ | **`NOT_ESTABLISHED`** |
| **Gate 3: Pure Compositional Reversal (`step=0`)** | `(1,0)+(0,1)` vs `(1,1)` at $q_{\text{pre}}=.1$ / $.9$ | **2.84 [2.01, 3.86] pp** / **0.87 [0.62, 1.17] pp** | Both $\ge 10.0$ pp, lower CIs $>0$, all 5 widths $>0$ | **`NOT_ESTABLISHED`** |
| **Gate 4: Compositional Remediation & Retention** | $q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$ (`step=1000`) | Recovery **95.10% [86.82, 100.41]**; control $\Delta$ **+0.02 [-0.10, +0.17] pp** | Recovery lo $\ge 50\%$, target $\ge 85\%$, control $\Delta \ge -5.0$ pp | **`NOT_ESTABLISHED`** |

---

## 2. Testing the Core Difficulty-Confound Thesis on Autoregressive Causal Transformers

All prior studies (Experiments 01–09) evaluated feedforward MLPs predicting all digits in parallel. Experiment 10 replicates the full Experiment 07 design on a **causal autoregressive transformer** (`AdditionTransformer`, 2 layers, 4 attention heads, causal self-attention, generating digits token-by-token from most to least significant).

When transformer checkpoints are pretrained across **heterogeneous carry-marginal matched histories** ($q_{\text{pre}} \in \{.1, .2, .5, .8, .9\}$) before balanced post-training ($q_{\text{post}}=0.50$):
- **`stable_base` joint held-out RMSE:** `0.2242 [0.0233, 0.3842]` bits/answer
- **`stable_oracle_history` joint held-out RMSE:** `0.2242 [0.0232, 0.3842]` bits/answer
- **`stable_difficulty` joint held-out RMSE:** `0.2241 [0.0230, 0.3840]` bits/answer
- **Relative RMSE reduction (`stable_base` $\to$ `stable_difficulty`):** **0.05% [0.03, 2.84]** (`NOT_ESTABLISHED`), also comparing against `stable_oracle_history` (**0.04% [0.03, 1.70]**).

---

## 3. Prequential (Online) MDL Code Length Along the Adaptation Stream

Measuring cumulative online prequential description length $\text{MDL}_{\text{preq}, s}(T)$ along the 1,000-step balanced adaptation stream for autoregressive transformers:
- **After $q_{\text{pre}}=0.10$ (single-carry heavy):** Online MDL is **0.0292 [0.0234, 0.0371]** bits/item on $S_{\text{iso}}$ vs **0.0527 [0.0402, 0.0687]** bits/item on $S_{\text{comp}}$ (excess adaptation cost **0.0235 [0.0165, 0.0320]** bits/item on $S_{\text{comp}}$, and **0.0234 [0.0175, 0.0298]** bits/item on pure `(1,1)` vs `(1,0)+(0,1)`).
- **After $q_{\text{pre}}=0.50$ (matched balanced history):** Online MDL is **0.0249 [0.0198, 0.0307]** bits/item on $S_{\text{iso}}$ vs **0.0285 [0.0228, 0.0351]** bits/item on $S_{\text{comp}}$.
- **After $q_{\text{pre}}=0.90$ (double-carry heavy):** Online MDL reverses to **0.0246 [0.0202, 0.0297]** bits/item on $S_{\text{iso}}$ vs **0.0207 [0.0167, 0.0251]** bits/item on $S_{\text{comp}}$ (excess adaptation cost **0.0039 [0.0029, 0.0051]** bits/item on $S_{\text{iso}}$, and **0.0021 [0.0005, 0.0035]** bits/item on pure `(1,0)+(0,1)` vs `(1,1)`).
- **Predicting Online Prequential MDL on Held-Out Scales & Histories:** Adding probed starting difficulty (`stable_difficulty` vs `stable_base`) reduces joint held-out prequential MDL RMSE from **0.1200 [0.0764, 0.1641]** to **0.1155 [0.0717, 0.1580]** bits/item—a **3.77% [1.21, 6.93]** reduction (`NOT_ESTABLISHED`).

---

## 4. Pure Compositional Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$)

- **Pre-intervention (`step=0`):** Single-carry control `(1,0)+(0,1)` accuracy is **99.90% [99.74, 99.99]** vs cascading two-carry target `(1,1)` accuracy **97.06% [95.89, 97.98]** (initial gap **2.84 [2.01, 3.86] pp**).
- **Post-remediation (`step=1000`):** Cascading two-carry `(1,1)` accuracy rises to **99.76% [99.59, 99.89]** (**+2.70 [+1.82, +3.83] pp**, recovering **95.10% [86.82, 100.41]** of the initial gap), while single-carry control `(1,0)+(0,1)` accuracy changes by **+0.02 [-0.10, +0.17] pp** to **99.92% [99.83, 99.98]** (`NOT_ESTABLISHED`).
