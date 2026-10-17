# Frozen Protocol: Reinforcement Learning with Verifiable Exact-Sum Rewards (RLVR) and Utility Scaling Laws (`08_rlvr_verifiable_reward_utility_laws`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/08_rlvr_verifiable_reward_utility_laws/PROTOCOL.md>

**TLDR:** Bridges `the-difficulty-confound` to Brando & Varun's reinforcement-learning environment scaling laws (`rl-4-code-verification` / [`predictability_bridge/README.md`](../ideas/predictability_bridge/README.md)) by replacing supervised post-training with **Reinforcement Learning with Verifiable 0/1 Exact-Sum Rewards (RLVR / leave-one-out GRPO, group size $G=16$)** across 200 checkpoints pretrained on heterogeneous carry-marginal matched histories ($q_{\text{pre}} \in \{0.10, 0.20, 0.50, 0.80, 0.90\}$).

## 1. Scientific Motivation

Across Experiments 01–07, all training interventions used supervised digit cross-entropy, requiring a standing caveat that the reinforcement-learning utility mechanism in [`experiments/ideas/predictability_bridge/README.md`](../ideas/predictability_bridge/README.md) remained untested:
> *"The proposed mechanism is **training history → starting-policy success probabilities → available reward signal and coverage → later learning utility**... For binary independent responses, a group of size $G$ contains both successes and failures with probability $1-p^G-(1-p)^G$; the existing experience module already implements this link."*

Experiment 08 directly tests this mechanism and utility law under exact answer matching and $50\%/50\%$ per-column carry-marginal invariance:
1. **Pure 0/1 Verifiable Exact-Sum Reward RL (`advance_rlvr`):**
   - For each prompt $x_b$ in a batch of $B=256$, the policy samples $G=16$ stochastic 4-digit candidates $\hat{y}_{b,1}, \dots, \hat{y}_{b,G} \sim \pi_\theta(\cdot \mid x_b)$ and receives **only binary exact-sum reward** $R_{b,g} = \mathbb{I}[\hat{y}_{b,g} == a_b + b_b] \in \{0, 1\}$.
   - Using the unbiased leave-one-out group advantage $\hat{A}_{b,g} = \frac{G}{G-1}(R_{b,g} - \bar{R}_b)$, any prompt whose $G$ rollouts are all correct ($\bar{R}_b = 1$) or all incorrect ($\bar{R}_b = 0$) receives $\hat{A}_{b,g} \equiv 0$, naturally concentrating gradient updates on items with non-zero usable signal density $\rho_i(G) = 1 - p_i^G - (1 - p_i)^G$ while protecting already-mastered items from interference.
2. **RL Improvement Utility Scaling Law (`rl-4-code-verification` Bridge):**
   - On an independent 1,440-item starting probe (`splits['probe']`), we measure starting loss $D_{m_0, s}$, pass@1 probability $p_{m_0, s}$, usable signal density $\rho_{m_0, s}(G) = \frac{1}{|S|}\sum_{i \in S} (1 - p_i^G - (1-p_i)^G)$, and pass@$G$ coverage $H_{m_0, s}(G) = \frac{1}{|S|}\sum_{i \in S} (1 - (1-p_i)^G)$.
   - We test whether adding these starting-checkpoint probe coordinates (`utility_signal_difficulty` and `utility_signal_coverage`) improves held-out forecasts of **RL improvement utility** $U = \text{Acc}_T - \text{Acc}_0$ across held-out widths (`96, 384`) and held-out pretraining histories (`0.20, 0.80`) compared with the scale+budget+structure baseline (`utility_base`).

## 2. Experimental Design

- **Seeds:** 8 fresh seeds (`8101, 8202, 8303, 8404, 8505, 8606, 8707, 8808`), split seed `20261028`.
- **Widths:** Fit `32, 64, 128`; held-out `96, 384`.
- **Pretraining Histories ($q_{\text{pre}}$):** Fit `0.10, 0.50, 0.90`; held-out `0.20, 0.80` (1,500 supervised updates, batch size 256).
- **RLVR Post-Training ($q_{\text{env}}=0.50$):** 1,000 RLVR updates (`checkpoints = [100, 300, 1000]`, $B=256$ prompts/step, $G=16$ rollouts/prompt, binary 0/1 exact-sum reward).

## 3. Prospective Pass/Fail Gates

1. **Gate 1 — Post-RLVR Conditional Loss Predictive Feature Value (`stable_difficulty` vs `stable_base`):**
   - `SUPPORTED` iff relative joint held-out RMSE reduction $\ge 10.0\%$, 95% seed-block bootstrap lower bound $> 0\%$, both held-out widths (`96` and `384`) improve, and 0 invalid forecasts occur.
2. **Gate 2 — RL Improvement Utility Scaling Law (`utility_signal_difficulty` vs `utility_base` on $U = \text{Acc}_T - \text{Acc}_0$):**
   - `SUPPORTED` iff relative joint held-out RMSE reduction on RL utility $U$ is $\ge 10.0\%$, 95% seed-block bootstrap lower bound $> 0\%$, and both held-out widths (`96` and `384`) improve.
3. **Gate 3 — RL Utility Reversal Across Pretraining Histories ($q_{\text{pre}}=0.10$ vs $q_{\text{pre}}=0.90$ at `step=1000`):**
   - `SUPPORTED` iff both `u_low` ($U_{s=2} - U_{s=1}$ at $q_{\text{pre}}=0.10$) and `u_high` ($U_{s=1} - U_{s=2}$ at $q_{\text{pre}}=0.90$) are $\ge 10.0$ percentage points with 95% lower bounds $> 0$ and positive point estimates at all 5 widths.
4. **Gate 4 — Pure 0/1 Verifiable-Reward RL Remediation & Strict No-Harm Retention ($q_{\text{pre}}=0.10 \to q_{\text{env}}=0.50$ at `step=1000`):**
   - `SUPPORTED` iff pure carry initial gap lower bound $\ge 15.0$ pp, recovery fraction lower bound $\ge 50.0\%$, and control single-carry `(1,0)+(0,1)` accuracy change estimate and 95% lower bound $\ge -5.00$ pp.
