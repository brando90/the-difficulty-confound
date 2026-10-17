# Frozen Protocol: Carry-Marginal Matched Heterogeneous Pretraining, No-Harm Remediation, and Prequential MDL (`07_prequential_mdl_and_length_transfer`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/07_prequential_mdl_and_length_transfer/PROTOCOL.md>

**TLDR:** Unifies Experiment 05 (heterogeneous pretraining histories + balanced remediation) and Experiment 06 (exact answer and 50%/50% per-column carry-marginal matched compositional regimes, input-digit TV `0.0398`), while recording cumulative **prequential (online) MDL code length** ([Voita & Titov, 2020](https://aclanthology.org/2020.emnlp-main.14/)) along the post-training trajectory.

## 1. Scientific Motivation

1. **Synthesizing Experiments 05 and 06:**
   - In [Experiment 06](../archive/06_compositional_carry_marginal_matched/RESULTS_compositional.md), holding exact answers and per-column carry rates ($50\%$ ones, $50\%$ tens) invariant proved that isolated single carries (`(1,0)`+`(0,1)`) and cascading two-carry chains (`(1,1)`) reverse difficulty rankings by **22.35 pp** and **17.74 pp**. However, because Experiment 06 used a shared `q=0.50` warmup, starting difficulty $D_{m_0}$ yielded only a **1.94% [-3.10, 6.50]** RMSE reduction.
   - By pretraining checkpoints across **heterogeneous carry-marginal matched histories** $q_{\text{pre}} \in \{0.10, 0.20, 0.50, 0.80, 0.90\}$ for 1,500 updates before balanced post-training ($q_{\text{post}}=0.50$), we test whether `stable_difficulty` achieves a $\ge 10\%$ joint held-out RMSE reduction **under invariant 50%/50% per-column carry marginals**.
2. **Prequential (Online) MDL Code Length:**
   - As noted in [`COMPLEXITY.md`](../archive/01_fixed_width_addition_exposure/COMPLEXITY.md), final conditional cross-entropy $L_{\theta_T}(y \mid x)$ omits the description length of learning the weights. During the 1,000 post-training steps, we record the cumulative online prequential code length $\text{MDL}_{\text{preq}}(T) = \sum_{t=1}^{T} L_{\theta_{t-1}}(\text{batch}_t)$ in bits per example to quantify the information cost of adapting from mismatched compositional histories ($q_{\text{pre}} \in \{0.1, 0.9\}$) versus matched history ($q_{\text{pre}}=0.5$).

## 2. Experimental Design

- **Seeds:** 8 fresh seeds (`7101, 7202, 7303, 7404, 7505, 7606, 7707, 7808`), split seed `20261027`.
- **Widths:** Fit `32, 64, 128`; held-out `96, 384`.
- **Pretraining Histories ($q_{\text{pre}}$):** Fit `0.10, 0.50, 0.90`; held-out `0.20, 0.80` (1,500 updates, batch size 256).
- **Post-Training Remediation ($q_{\text{post}}$):** Balanced `0.50` across all 4 carry cells (checkpoints `100, 300, 1000` updates).
- **Invariants:** 180 eligible answers, 7,200 training pairs/mixture (40/answer), exact 50.00% ones carry (`3600/7200`), 50.00% tens carry (`3600/7200`), 0.00% hundreds carry across all mixtures.

## 3. Prospective Pass/Fail Gates

1. **Gate 1 — Starting-Difficulty Predictive Feature Value under Carry-Marginal Matched Heterogeneous Histories (`stable_difficulty` vs `stable_base`):**
   - `SUPPORTED` iff relative joint held-out RMSE reduction $\ge 10.0\%$, 95% seed-block bootstrap lower bound $> 0\%$, both held-out widths (`96` and `384`) improve, and 0 invalid forecasts occur.
2. **Gate 2 — Pure Compositional Pre-Intervention Ranking Reversal (`(1,0)+(0,1)` vs `(1,1)` at `step=0`):**
   - `SUPPORTED` iff both `pure_low` ($q_{\text{pre}}=0.1$) and `pure_high` ($q_{\text{pre}}=0.9$) are $\ge 10.0$ percentage points with 95% bootstrap lower bounds $> 0$ and positive point estimates at all 5 widths.
3. **Gate 3 — Targeted Compositional Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.1 \to q_{\text{post}}=0.5$ at `step=1000`):**
   - `SUPPORTED` iff initial gap lower bound $\ge 15.0$ pp, recovery fraction lower bound $\ge 50.0\%$, post-training target accuracy $\ge 85.0\%$, and control accuracy change estimate and 95% lower bound $\ge -5.00$ pp.
