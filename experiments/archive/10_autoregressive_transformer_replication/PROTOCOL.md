# Protocol: Autoregressive Transformer Architecture Generalization (`10_autoregressive_transformer_replication`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/10_autoregressive_transformer_replication/PROTOCOL.md>

**TLDR:** Replicates Experiment 07's carry-marginal matched heterogeneous pretraining, balanced remediation, and prequential (online) MDL code length tracking on an **autoregressive causal transformer** (`AdditionTransformer`, 2 layers, 4 attention heads, causal self-attention, generating digits token-by-token from most to least significant).

## 1. Scientific Motivation

1. **Architecture Generalization (From MLPs to Autoregressive Transformers):**
   - Experiments 01–09 established that apparent task difficulty rankings reverse completely under controlled exposure shifts and that independently probed starting difficulty $D_{m_0}$ reduces held-out forecasting RMSE by up to $47.20\%$. However, all prior studies used multi-layer perceptrons (MLPs) that predicted all output digits in parallel from concatenated one-hot input features.
   - Modern foundation models and LLMs are **autoregressive causal transformers** that attend over input prompt tokens and generate output tokens sequentially.
   - Experiment 10 tests whether the core findings of this project—(1) pure compositional difficulty ranking reversal under carry-marginal matching, (2) prequential online MDL adaptation cost reversal, (3) predictive feature value of probed starting difficulty under heterogeneous histories, and (4) effective remediation without harm to existing capabilities—hold on an autoregressive transformer.

2. **Causal Transformer Formulation:**
   - **Prompt tokens:** $a_2, a_1, a_0, +, b_2, b_1, b_0, =$ (8 input tokens).
   - **Target tokens:** $y_3, y_2, y_1, y_0$ (4 output tokens, most-significant digit first).
   - **Attention:** Standard causal multi-head self-attention with learned positional embeddings.
   - **Loss:** Teacher-forced cross-entropy over the 4 output positions.
   - **Inference:** Greedy autoregressive token generation.

## 2. Experimental Design

- **Seeds:** 8 fresh seeds (`10101, 10202, 10303, 10404, 10505, 10606, 10707, 10808`), split seed `20261031`, analysis seed `31102026`.
- **Widths:** Fit `32, 64, 128`; held-out `96, 384` (embedding dimensions $d_{\text{model}}$).
- **Pretraining Histories ($q_{\text{pre}}$):** Fit `0.10, 0.50, 0.90`; held-out `0.20, 0.80` (1,500 updates, batch size 256).
- **Post-Training Remediation ($q_{\text{post}}$):** Balanced `0.50` across all 4 carry cells (checkpoints `100, 300, 1000` updates).
- **Invariants:** 180 eligible answers, 7,200 training pairs/mixture (40/answer), exact 50.00% ones carry (`3600/7200`), 50.00% tens carry (`3600/7200`), 0.00% hundreds carry across all mixtures (input-digit TV `0.0398`).

## 3. Prospective Pass/Fail Gates

1. **Gate 1 — Starting-Difficulty Predictive Feature Value on Held-Out Conditional Loss (`stable_difficulty` vs `stable_base`):**
   - `SUPPORTED` iff relative joint held-out RMSE reduction $\ge 10.0\%$, 95% seed-block bootstrap lower bound $> 0\%$, both held-out widths (`96` and `384`) improve, and 0 invalid forecasts occur.
2. **Gate 2 — Prequential Online MDL Reversal & Predictive Feature Value:**
   - **Prequential MDL Reversal:** `SUPPORTED` iff online adaptation code length reverses direction between $q_{\text{pre}}=0.10$ and $q_{\text{pre}}=0.90$ (`pure_mdl_low > 0`, `pure_mdl_high > 0` with 95% lower bounds $> 0$ and positive point estimates at all 5 widths).
   - **Prequential MDL Prediction (`stable_difficulty` vs `stable_base` on `prequential_mean_bits`):** `SUPPORTED` iff relative joint held-out RMSE reduction $\ge 10.0\%$ with 95% lower bound $> 0\%$.
3. **Gate 3 — Pure Compositional Pre-Intervention Ranking Reversal (`(1,0)+(0,1)` vs `(1,1)` at `step=0`):**
   - `SUPPORTED` iff both `pure_low` ($q_{\text{pre}}=0.1$) and `pure_high` ($q_{\text{pre}}=0.9$) are $\ge 10.0$ percentage points with 95% bootstrap lower bounds $> 0$ and positive point estimates at all 5 widths.
4. **Gate 4 — Targeted Compositional Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.1 \to q_{\text{post}}=0.5$ at `step=1000`):**
   - `SUPPORTED` iff pure carry initial gap lower bound $\ge 15.0$ pp, recovery fraction lower bound $\ge 50.0\%$, post-training target `(1,1)` accuracy $\ge 85.0\%$, and control `(1,0)+(0,1)` accuracy change estimate and 95% lower bound $\ge -5.00$ pp.
