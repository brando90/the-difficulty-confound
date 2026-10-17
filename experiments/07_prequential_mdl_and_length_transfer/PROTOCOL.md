# Frozen Protocol: Carry-Marginal Matched Heterogeneous Pretraining, No-Harm Remediation, and Prequential Online MDL (`07_prequential_mdl_and_length_transfer`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/07_prequential_mdl_and_length_transfer/PROTOCOL.md>

**TLDR:** Unifies Experiment 05 (heterogeneous pretraining histories + balanced remediation) and Experiment 06 (exact answer and 50%/50% per-column carry-marginal matched compositional regimes, input-digit TV `0.0398`), while recording exact stratum- and cell-resolved **prequential (online) MDL code length** ([Voita & Titov, 2020](https://aclanthology.org/2020.emnlp-main.14/)) along the post-training adaptation trajectory.

## 1. Scientific Motivation

1. **Synthesizing Experiments 05 and 06 (Carry-Marginal Matched Heterogeneous Histories):**
   - In [Experiment 06](../archive/06_compositional_carry_marginal_matched/RESULTS_compositional.md), holding exact answers and per-column carry rates ($50\%$ ones, $50\%$ tens) invariant proved that isolated single carries (`(1,0)`+`(0,1)`) and cascading two-carry chains (`(1,1)`) reverse difficulty rankings by **22.35 pp** and **17.74 pp**. However, because Experiment 06 used a shared `q=0.50` warmup, starting difficulty $D_{m_0}$ yielded only a **1.94% [-3.10, 6.50]** RMSE reduction.
   - By pretraining checkpoints across **heterogeneous carry-marginal matched histories** $q_{\text{pre}} \in \{0.10, 0.20, 0.50, 0.80, 0.90\}$ for 1,500 updates before balanced post-training ($q_{\text{post}}=0.50$), we test whether `stable_difficulty` achieves a $\ge 10\%$ joint held-out RMSE reduction **under invariant 50%/50% per-column carry marginals**.
2. **Closing the Kolmogorov / Prequential Online MDL Gap (`COMPLEXITY.md`):**
   - As noted in [`COMPLEXITY.md`](../archive/01_fixed_width_addition_exposure/COMPLEXITY.md), final conditional cross-entropy $L_{\theta_T}(y \mid x)$ omits the description length of learning the weights. During the 1,000 balanced post-training steps ($q_{\text{post}}=0.50$), we record the exact per-stratum and per-cell online prequential code length $\text{MDL}_{\text{preq}, s}(T) = \frac{1}{N_s(T)} \sum_{t=1}^{T} \sum_{i \in B_t \cap s} L_{\theta_{t-1}}(y_i \mid x_i)$ in bits per example *before* each gradient update, quantifying both the online adaptation description-length reversal and whether `stable_difficulty` predicts online prequential MDL across held-out widths and mixtures.

## 2. Experimental Design

- **Seeds:** 8 fresh seeds (`7101, 7202, 7303, 7404, 7505, 7606, 7707, 7808`), split seed `20261027`.
- **Widths:** Fit `32, 64, 128`; held-out `96, 384`.
- **Pretraining Histories ($q_{\text{pre}}$):** Fit `0.10, 0.50, 0.90`; held-out `0.20, 0.80` (1,500 updates, batch size 256).
- **Post-Training Remediation ($q_{\text{post}}$):** Balanced `0.50` across all 4 carry cells (checkpoints `100, 300, 1000` updates).
- **Invariants:** 180 eligible answers, 7,200 training pairs/mixture (40/answer), exact 50.00% ones carry (`3600/7200`), 50.00% tens carry (`3600/7200`), 0.00% hundreds carry across all mixtures.

## 3. Prospective Pass/Fail Gates

1. **Gate 1 — Starting-Difficulty Predictive Feature Value on Held-Out Conditional Loss (`stable_difficulty` vs `stable_base`):**
   - `SUPPORTED` iff relative joint held-out RMSE reduction $\ge 10.0\%$, 95% seed-block bootstrap lower bound $> 0\%$, both held-out widths (`96` and `384`) improve, and 0 invalid forecasts occur.
2. **Gate 2 — Prequential Online MDL Reversal & Predictive Feature Value:**
   - **Prequential MDL Reversal:** `SUPPORTED` iff online adaptation code length reverses direction between $q_{\text{pre}}=0.10$ and $q_{\text{pre}}=0.90$ (`mdl_low > 0`, `mdl_high > 0`, `pure_mdl_low > 0`, `pure_mdl_high > 0` with 95% lower bounds $> 0$ and positive point estimates at all 5 widths).
   - **Prequential MDL Prediction (`stable_difficulty` vs `stable_base` on `prequential_mean_bits`):** `SUPPORTED` iff relative joint held-out RMSE reduction $\ge 10.0\%$ with 95% lower bound $> 0\%$.
3. **Gate 3 — Pure Compositional Pre-Intervention Ranking Reversal (`(1,0)+(0,1)` vs `(1,1)` at `step=0`):**
   - `SUPPORTED` iff both `pure_low` ($q_{\text{pre}}=0.1$) and `pure_high` ($q_{\text{pre}}=0.9$) are $\ge 10.0$ percentage points with 95% bootstrap lower bounds $> 0$ and positive point estimates at all 5 widths.
4. **Gate 4 — Targeted Compositional Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.1 \to q_{\text{post}}=0.5$ at `step=1000`):**
   - `SUPPORTED` iff pure carry initial gap lower bound $\ge 15.0$ pp, recovery fraction lower bound $\ge 50.0\%$, post-training target `(1,1)` accuracy $\ge 85.0\%$, and control `(1,0)+(0,1)` accuracy change estimate and 95% lower bound $\ge -5.00$ pp.
