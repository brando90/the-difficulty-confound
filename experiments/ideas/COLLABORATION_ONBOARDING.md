# Collaborator Onboarding & Next PRs (`the-difficulty-confound`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/ideas/COLLABORATION_ONBOARDING.md>

Welcome to the project, Laolu (`Laolu24`)! This guide summarizes what we have already established across Experiments 01–08 and how the reproducible experiment pipeline works. Pull requests #1–#7 are already merged. A second look is open in [issue #8](https://github.com/brando90/the-difficulty-confound/issues/8).

---

## 1. The Core Thesis in One Paragraph

**Is a benchmark task intrinsically hard, or has the model simply not practiced the right distribution yet?**
When we evaluate a model on a fixed skill (here, exact 3-digit integer addition with an unchanged ground-truth algorithm and unchanged test items), apparent difficulty rankings between "easy" (no-carry or single-carry) and "hard" (cascading two-carry) problems **reverse completely** depending on training exposure—across final accuracy, **prequential (online) MDL code length**, and **verifiable-reward RL improvement utility ($U = \text{Acc}_T - \text{Acc}_0$)**—even when we hold the exact answer histogram **and** the exact per-column carry frequencies ($50\%$ ones, $50\%$ tens) fixed across every training condition. Furthermore, when we want to **predict** held-out post-training performance across unseen model scales and unseen mixtures, an independent starting-difficulty probe ($D_{m_0}$) adds almost nothing ($1.94\%\text{–}3.48\%$) when all checkpoints share the same warmup history, but reduces held-out conditional loss RMSE by **47.20%** (Expt 05), **36.22%** (Expt 07), and **39.87%** (Expt 08 under RLVR)—plus **56.93%** on prequential online MDL (Expt 07)—when checkpoints arrive with **heterogeneous prior training histories**. Finally, balanced post-training in `experiments/archive/05_pretraining_history_and_retention/` passes its strict no-harm retention gate (no-carry control **+1.09 [+0.33, +1.89] percentage points**, two-carry accuracy **90.70% [88.79, 92.43]**). In `experiments/archive/07_prequential_mdl_and_length_transfer/`, the single-carry control also improves (**+5.13 [+4.13, +6.38] percentage points**) and gap recovery is **104.29% [101.45, 107.53]**, but Gate 4 is **NOT_ESTABLISHED**: pooled cascading-two-carry accuracy is **81.09% [79.51, 82.88]**, below the prespecified 85% line (width 32 is **27.80%** after remediation; width 64 is **82.26%**). In `experiments/archive/08_rlvr_verifiable_reward_utility_laws/`, the verifiable-reward remediation gate is **SUPPORTED** (control **+6.45 [+4.69, +7.91] percentage points**). Its pooled additive utility law is **NOT_ESTABLISHED** (**-15.78% [-35.91, 3.15]**).

---

## 2. Summary of Completed Experiments (01–08)

All completed experiments live in [`experiments/archive/`](../archive/README.md) with every per-item prediction, saved weight `.pt`, manifest, SHA-256 receipt, and independent replay verifier committed:

| Expt | Directory | Question Tested | Main Outcome |
|---|---|---|---|
| **01** | [`01_fixed_width_addition_exposure`](../archive/01_fixed_width_addition_exposure/RESULTS_addition_exposure.md) | Does training exposure close the 3-carry gap on fixed 3-digit addition? | **94.34% [92.74, 95.86]** of gap recovered (`9.60% -> 94.89%`), but no-carry control dropped **6.08 pp** (missed strict $\le 5.0$ pp retention gate). |
| **02** | [`02_heldout_exposure_difficulty_laws`](../archive/02_heldout_exposure_difficulty_laws/RESULTS_heldout_laws.md) | Does an additive alignment-scaling law with exposure + starting difficulty improve held-out forecasts? | Primary combined law **NOT_ESTABLISHED**: RMSE reduction **6.80% [-3.52, 14.80]** (below the 10% gate) with **64/384** negative-loss forecasts. Difficulty-only ablation improved **15.80% [10.27, 20.52]** but produced **52/384** negative-loss forecasts. |
| **03** | [`03_answer_matched_alignment`](../archive/03_answer_matched_alignment/RESULTS_matched_alignment.md) | Does difficulty ranking still reverse when training & test sets have identical answer histograms? | **Ranking reversal SUPPORTED** (**25.30 pp** and **24.58 pp** across all widths). Unconstrained positive log-linear difficulty law blew up on held-out width 256 (`292.79` bits RMSE). |
| **04** | [`04_monotone_ridge_heldout_addition`](../archive/04_monotone_ridge_heldout_addition/RESULTS_stable.md) | Does a constrained monotone ridge law (`stable_difficulty`) repair extrapolation blow-up on fresh seeds & width 384? | **Extrapolation repair SUPPORTED** (`1.94e12 -> 0.4835` bits, **100.00% [99.90, 100.00]** reduction, 0 invalid forecasts) and ranking reversal replicated (**24.07 / 24.48 pp**). Starting-difficulty feature gate **NOT_ESTABLISHED**: under shared warmup (`q=0.50`), `stable_difficulty` adds only **3.48% [1.12, 6.24]** over `stable_base`, below the 10% line. |
| **05** | [`05_pretraining_history_and_retention`](../archive/05_pretraining_history_and_retention/RESULTS_history_retention.md) | What happens when starting checkpoints have **heterogeneous pretraining histories** ($q_{\text{pre}} \in \{.1,.2,.5,.8,.9\}$) before balanced post-training ($q_{\text{post}}=.5$)? | **Forecast, retention, and ranking reversal SUPPORTED. Extrapolation repair NOT_ESTABLISHED** (`stable_difficulty` vs `legacy_difficulty` **-0.82% [-1.11, -0.57]**). `stable_difficulty` cuts held-out RMSE by **47.20% [42.88, 53.08]** (`0.1902 -> 0.1004` bits) and beats oracle history (`0.1580`) by **36.43%**. Post-training recovers **87.55% [83.60, 90.72]** of the $q_{\text{pre}}=0.1$ carry gap with **+1.09 [+0.33, +1.89] pp** no-carry retention. |
| **06** | [`06_compositional_carry_marginal_matched`](../archive/06_compositional_carry_marginal_matched/RESULTS_compositional.md) | Does the ranking reversal survive when **both exact answers and per-column carry rates (50% ones, 50% tens)** are held invariant across every mixture? | **Pure compositional reversal SUPPORTED:** input-digit TV mismatch cut $>11\times$ (`0.443 -> 0.0398`), yet isolated single carries lead by **22.35 [20.60, 24.00] pp** at $q=.1$ and cascading two-carry chains lead by **17.74 [14.62, 21.12] pp** at $q=.9$. Shared-warmup difficulty feature is **NOT_ESTABLISHED** (**1.94% [-3.10, 6.50]**), the contrast with Experiment 05's heterogeneous-history gain. |
| **07** | [`07_prequential_mdl_and_length_transfer`](../archive/07_prequential_mdl_and_length_transfer/RESULTS_mdl_history.md) | Does **prequential (online) MDL code length** reverse across heterogeneous histories under $50\%/50\%$ carry-marginal invariance, and does `stable_difficulty` predict both final loss and online MDL? | **Loss law, MDL reversal, and MDL law SUPPORTED. Gate 4 NOT_ESTABLISHED.** `stable_difficulty` cuts loss RMSE by **36.22% [32.45, 40.51]** and online MDL RMSE by **56.93% [55.01, 58.67]**; online MDL reverses (`0.4423` vs `0.1773` bits/item). Gap recovery is **104.29% [101.45, 107.53]** with control **+5.13 [+4.13, +6.38] pp**, but pooled `(1,1)` accuracy is **81.09% [79.51, 82.88]** (prespecified line 85%). Width-32 eight-seed mean is **27.80%**; width 64 is **82.26%**. |
| **08** | [`08_rlvr_verifiable_reward_utility_laws`](../archive/08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md) | Does pure 0/1 verifiable-reward RL (RLVR / leave-one-out GRPO, $G=16$) exhibit history-dependent utility reversal, natural no-harm retention, and predictable RL improvement utility $U = \text{Acc}_T - \text{Acc}_0$? | **Post-RLVR loss (`39.87%`), utility reversal (`29.07` / `32.70` pp), and RLVR remediation (`78.23%` recovery, `+6.45 pp` retention) SUPPORTED.** Additive linear RL utility law cuts width-96 RMSE by **48.56%** (`14.59 -> 7.50` pp) but overshoots remaining headroom $1 - \text{Acc}_0$ at near-ceiling width 384 (`-15.78%` pooled). |

---

## 3. Repository Workflow & Scientific Hygiene Rules

Every experiment follows a strict 5-step workflow so that neither human nor agent can tune post-hoc thresholds:

1. **Freeze `PROTOCOL.md` before running:** State the exact research question, fit/held-out splits, mathematical functional forms, and numerical pass/fail gates (`SUPPORTED` vs `NOT_ESTABLISHED`) *before* training models.
2. **Preflight `--check-only` & unit tests:**
   - `python3 experiments/archive/<id>/expt_v1/test_laws.py` verifies law constraints and independent cross-fitter parity.
   - `python3 experiments/archive/<id>/expt_v1/run.py --check-only` verifies exact dataset cardinalities, train/probe/eval disjointness, and marginal invariants without training.
3. **Run & Replay-Verify (`verify.py`):**
   - Training saves every checkpoint's `.pt` weights and per-item predictions.
   - `python3 experiments/archive/<id>/expt_v1/verify.py` reloads every saved checkpoint, recomputes forward passes independently, checks greedy accuracy against saved tables, and requires bit-loss differences `<= 1e-6`. Completed studies are under `experiments/archive/`, not `experiments/<id>/`. For Experiments 05–08 the verifier falls back to the sibling copy of `experiments/archive/02_heldout_exposure_difficulty_laws/` after the archive move, and that path fallback is the only source file allowed to differ from the training hash. On torch 2.7.0 the greedy predictions still match and the bit differences are about 1e-5, so the frozen `1e-6` assert fails even though the training receipt recorded `0.0` on torch 2.5.1.
4. **Seed-Block Bootstrap Uncertainty (`analyze.py` & `report.py`):**
   - Never treat individual test items or checkpoints from the same seed as independent! All 95% confidence intervals use paired bootstrap resampling over the **8 independent seed blocks** (`20,000` replicates, the `bootstrap_draws` value in each experiment config).
5. **Retain All Counterevidence:**
   - When a gate is `NOT_ESTABLISHED`, report it with the same prominence as `SUPPORTED` gates. Never drop outlier seeds or failed forecasts.

---

## 4. Next experiment

Pull requests #1–#7 are merged. The second-look notes are in [issue #8](https://github.com/brando90/the-difficulty-confound/issues/8).

Experiment 08 revealed a sharp, actionable finding for the RL environment scaling law (`rl-4-code-verification`):
- At **interpolated held-out width 96**, adding stochastic rollout signal density $\log \rho_{m_0}$, headroom $H_{m_0}$, and $\log D_{m_0}$ reduced RL improvement utility RMSE by **48.56% [41.65, 57.30]** (`14.59 -> 7.50` pp), beating even oracle pretraining history (`8.53` pp) by **12.01%**.
- At **extrapolated near-ceiling width 384**, where starting accuracy $\text{Acc}_0$ is already $78\%\text{–}99.8\%$, the **additive linear** utility formula $\widehat{U}_{\text{add}} = \text{clip}(b_0 + b_N x_N + s(C)(\dots), 0, 1)$ overshot remaining headroom $1 - \text{Acc}_0$ (`19.71 -> 27.38` pp), whereas the multiplicative log-loss law `stable_difficulty` succeeded at both widths (**39.87%** joint reduction).

This directly motivates **Experiment 09 (`09_headroom_bounded_rlvr_utility_law`)**: replacing the additive linear RL utility formula with a **headroom-bounded fractional-recovery RL utility law** $\widehat{U}_{\text{bounded}} = (1 - \text{Acc}_0) \cdot (1 - \exp(-r_\theta(N, C, c, \rho_{m_0}, D_{m_0})))$ that structurally guarantees $0 \le \widehat{U} \le 1 - \text{Acc}_0$ across all model scales!
