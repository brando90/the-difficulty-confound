# Protocol: Calibrated Pretraining Horizon for Causal Autoregressive Transformers (`11_transformer_calibrated_pretraining_horizon`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/11_transformer_calibrated_pretraining_horizon/PROTOCOL.md>

**TLDR:** Calibrates the pretraining horizon of the causal autoregressive transformer (`AdditionTransformer`, 2 layers, 4 heads) from 1,500 to 300 steps ($T_{\text{pre}}=300$) to match the active learning regime of MLPs, restoring active headroom across all five widths ($d_{\text{model}} \in \{32, 64, 128, 96, 384\}$) and evaluating whether starting-difficulty scaling laws predict held-out loss and prequential MDL on transformers.

## 1. Scientific Motivation

1. **Resolving Experiment 10's Ceiling Saturation:**
   - In Experiment 10, causal autoregressive transformers trained for 1,500 updates on arithmetic tokens reached near-ceiling accuracy (>99.8% to 100.0%, loss < 0.005 bits) for all widths $\ge 64$.
   - While width 32 showed a large +11.77 pp difficulty gap and 94.52% remediation, the lack of headroom at widths 64–384 compressed pooled error reduction to near-zero (0.05%), leaving all prospective gates `NOT_ESTABLISHED`.
   - This established a critical fact: transformers learn digit addition far more sample-efficiently than MLPs (~5x faster).
   - Experiment 11 calibrates the pretraining horizon to $T_{\text{pre}} = 300$ updates (76,800 token sequences; ~10 epochs), capturing models during their active learning transition where difficulty gaps and error headroom exist across all scales.

2. **Causal Transformer Architecture & Invariants:**
   - **Architecture:** 2 layers, 4 attention heads, causal self-attention, token-by-token greedy generation ($a_2 a_1 a_0 + b_2 b_1 b_0 = y_3 y_2 y_1 y_0$).
   - **Data Invariance:** 180 eligible answers, 7,200 training pairs per mixture (40/answer), exact 50.00% ones carry and 50.00% tens carry across all mixtures (input-digit TV `0.0398`).

## 2. Experimental Design

- **Seeds:** 8 fresh paired seeds (`11101, 11202, 11303, 11404, 11505, 11606, 11707, 11808`), split seed `20261111`, analysis seed `11112026`.
- **Widths:** Fit `32, 64, 128`; held-out `96, 384` ($d_{\text{model}}$).
- **Pretraining Histories ($q_{\text{pre}}$):** Fit `0.10, 0.50, 0.90`; held-out `0.20, 0.80` (calibrated to **300 updates**, batch size 256).
- **Post-Training Remediation ($q_{\text{post}}$):** Balanced `0.50` across all 4 carry cells (checkpoints `100, 300, 1000` updates).

## 3. Prospective Pass/Fail Gates

1. **Gate 1 — Starting-Difficulty Predictive Feature Value on Held-Out Conditional Loss (`stable_difficulty` vs `stable_base`):**
   - `SUPPORTED` iff relative joint held-out RMSE reduction $\ge 10.0\%$, 95% seed-block bootstrap lower bound $> 0\%$, both held-out widths (`96` and `384`) improve, and 0 invalid forecasts occur.
2. **Gate 2a — Prequential Online MDL Reversal:**
   - `SUPPORTED` iff online adaptation code length reverses direction between $q_{\text{pre}}=0.10$ and $q_{\text{pre}}=0.90$ (`pure_mdl_low > 0`, `pure_mdl_high > 0` with 95% lower bounds $> 0$ and positive point estimates at all 5 widths).
3. **Gate 2b — Prequential Online MDL Prediction (`stable_difficulty` vs `stable_base`):**
   - `SUPPORTED` iff relative joint held-out prequential RMSE reduction $\ge 10.0\%$ with 95% lower bound $> 0\%$.
4. **Gate 3 — Pure Compositional Pre-Intervention Ranking Reversal (`(1,0)+(0,1)` vs `(1,1)` at `step=0`):**
   - `SUPPORTED` iff both `pure_low` ($q_{\text{pre}}=0.1$) and `pure_high` ($q_{\text{pre}}=0.9$) are $\ge 10.0$ percentage points with 95% bootstrap lower bounds $> 0$ and positive point estimates at all 5 widths.
5. **Gate 4 — Targeted Compositional Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.1 \to q_{\text{post}}=0.5$ at `step=1000`):**
   - `SUPPORTED` iff pure carry initial gap lower bound $\ge 15.0$ pp, recovery fraction lower bound $\ge 50.0\%$, post-training target `(1,1)` accuracy $\ge 85.0\%$, and control `(1,0)+(0,1)` accuracy change estimate and 95% lower bound $\ge -5.00$ pp.
