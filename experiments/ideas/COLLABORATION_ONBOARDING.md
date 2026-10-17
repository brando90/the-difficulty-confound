# Collaborator Onboarding & Next PRs (`the-difficulty-confound`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/ideas/COLLABORATION_ONBOARDING.md>

Welcome to the project, Laolu (`Laolu24`)! This guide summarizes what we have already established across Experiments 01–06, how our reproducible experiment pipeline works, and the open PRs / next experiments you can jump into right away.

---

## 1. The Core Thesis in One Paragraph

**Is a benchmark task intrinsically hard, or has the model simply not practiced the right distribution yet?**
When we evaluate a model on a fixed skill (here, exact 3-digit integer addition with an unchanged ground-truth algorithm and unchanged test items), apparent difficulty rankings between "easy" (no-carry or single-carry) and "hard" (cascading two-carry) problems **reverse completely** depending on training exposure—even when we hold the exact answer histogram **and** the exact per-column carry frequencies ($50\%$ ones, $50\%$ tens) fixed across every training condition. Furthermore, when we want to **predict** held-out post-training loss across unseen model scales and unseen mixtures, an independent starting-difficulty probe ($D_{m_0}$) adds almost nothing ($1.94\%\text{–}3.48\%$) when all checkpoints share the same warmup history, but reduces held-out RMSE by **47.20% [42.88, 53.08]** (even beating an oracle pretraining-mixture law by **36.43%**) when checkpoints arrive with **heterogeneous prior training histories**. Finally, balanced post-training recovers **87.55% [83.60, 90.72]** of a pretraining-induced carry gap while **improving** control no-carry accuracy by **+1.09 [+0.33, +1.89] percentage points** (passing the strict no-harm retention gate).

---

## 2. Summary of Completed Experiments (01–06)

All completed experiments live in [`experiments/archive/`](../archive/README.md) with every per-item prediction, saved weight `.npz`, manifest, SHA-256 receipt, and independent replay verifier committed:

| Expt | Directory | Question Tested | Main Outcome |
|---|---|---|---|
| **01** | [`01_fixed_width_addition_exposure`](../archive/01_fixed_width_addition_exposure/RESULTS_addition_exposure.md) | Does training exposure close the 3-carry gap on fixed 3-digit addition? | **94.34% [92.74, 95.86]** of gap recovered (`9.60% -> 94.89%`), but no-carry control dropped **6.08 pp** (missed strict $\le 5.0$ pp retention gate). |
| **02** | [`02_heldout_exposure_difficulty_laws`](../archive/02_heldout_exposure_difficulty_laws/RESULTS_heldout_laws.md) | Does an additive alignment-scaling law with exposure + starting difficulty improve held-out forecasts? | Combined law reduced RMSE **6.80% [-3.52, 14.80]** (below 10% gate); difficulty-only ablation improved **15.80% [10.27, 20.52]** but produced 52/384 negative-loss forecasts. |
| **03** | [`03_answer_matched_alignment`](../archive/03_answer_matched_alignment/RESULTS_matched_alignment.md) | Does difficulty ranking still reverse when training & test sets have identical answer histograms? | **Ranking reversal SUPPORTED** (**25.30 pp** and **24.58 pp** across all widths). Unconstrained positive log-linear difficulty law blew up on held-out width 256 (`292.79` bits RMSE). |
| **04** | [`04_monotone_ridge_heldout_addition`](../archive/04_monotone_ridge_heldout_addition/RESULTS_stable.md) | Does a constrained monotone ridge law (`stable_difficulty`) repair extrapolation blow-up on fresh seeds & width 384? | **Extrapolation repair SUPPORTED** (`1.94e12 -> 0.4835` bits, **100.00% [99.90, 100.00]** reduction, 0 invalid forecasts) and ranking reversal replicated (**24.07 / 24.48 pp**). Under shared warmup (`q=0.50`), starting difficulty added only **3.48% [1.12, 6.24]** over `stable_base`. |
| **05** | [`05_pretraining_history_and_retention`](../archive/05_pretraining_history_and_retention/RESULTS_history_retention.md) | What happens when starting checkpoints have **heterogeneous pretraining histories** ($q_{\text{pre}} \in \{.1,.2,.5,.8,.9\}$) before balanced post-training ($q_{\text{post}}=.5$)? | **ALL GATES SUPPORTED:** (1) `stable_difficulty` cuts held-out RMSE by **47.20% [42.88, 53.08]** (`0.1902 -> 0.1004` bits) and beats oracle history (`0.1580`) by **36.43%**; (2) post-training recovers **87.55% [83.60, 90.72]** of the $q_{\text{pre}}=0.1$ carry gap with **+1.09 [+0.33, +1.89] pp** no-carry retention. |
| **06** | [`06_compositional_carry_marginal_matched`](../archive/06_compositional_carry_marginal_matched/RESULTS_compositional.md) | Does the ranking reversal survive when **both exact answers and per-column carry rates (50% ones, 50% tens)** are held invariant across every mixture? | **Pure compositional reversal SUPPORTED:** input-digit TV mismatch cut $>11\times$ (`0.443 -> 0.0398`), yet isolated single carries lead by **22.35 [20.60, 24.00] pp** at $q=.1$ and cascading two-carry chains lead by **17.74 [14.62, 21.12] pp** at $q=.9$. Shared-warmup difficulty gain is **1.94% [-3.10, 6.50]**, confirming Expt 05's contrast. |

---

## 3. Repository Workflow & Scientific Hygiene Rules

Every experiment follows a strict 5-step workflow so that neither human nor agent can tune post-hoc thresholds:

1. **Freeze `PROTOCOL.md` before running:** State the exact research question, fit/held-out splits, mathematical functional forms, and numerical pass/fail gates (`SUPPORTED` vs `NOT_ESTABLISHED`) *before* training models.
2. **Preflight `--check-only` & unit tests:**
   - `python3 experiments/<id>/expt_v1/test_laws.py` verifies law constraints and independent cross-fitter parity.
   - `python3 experiments/<id>/expt_v1/run.py --check-only` verifies exact dataset cardinalities, train/probe/eval disjointness, and marginal invariants without training.
3. **Run & Replay-Verify (`verify.py`):**
   - Training saves every checkpoint's `.npz` weights and per-item predictions.
   - `python3 experiments/<id>/expt_v1/verify.py` reloads every `.npz` checkpoint from disk, recomputes forward passes independently, checks exact bit loss (`max_abs_diff < 1e-9`) and greedy accuracy against saved tables, and independently refits all scaling laws.
4. **Seed-Block Bootstrap Uncertainty (`analyze.py` & `report.py`):**
   - Never treat individual test items or checkpoints from the same seed as independent! All 95% confidence intervals use paired bootstrap resampling over the **8 independent seed blocks** (`10,000` replicates).
5. **Retain All Counterevidence:**
   - When a gate is `NOT_ESTABLISHED`, report it with the same prominence as `SUPPORTED` gates. Never drop outlier seeds or failed forecasts.

---

## 4. Open PRs & Next Experiments Ready for Laolu

We have prepared an active PR branch (`exp07-prequential-mdl-and-length-transfer`) with a scaffolded protocol and preflight checks so you can dive right in and help close it!

### Active Starter PR: Experiment 07 — Prequential MDL Code Length & Compositional Length Transfer (`07_prequential_mdl_and_length_transfer`)
- **Why this matters:**
  1. **Closing the Kolmogorov / MDL gap (`COMPLEXITY.md`):** Experiments 01–06 measured *final conditional predictive code length* (bits/answer after training), which excludes the description length of learning the weights. Following [Voita & Titov (2020)](https://aclanthology.org/2020.emnlp-main.14/), **prequential (online) MDL code length** $\sum_{t} L_{\theta_{t-1}}(y_t \mid x_t)$ charges for every bit spent along the learning trajectory as the model absorbs examples.
  2. **Combined Heterogeneous History + Carry-Marginal Matched Reversal:** Combining Experiment 05's heterogeneous pretraining histories ($q_{\text{pre}}$) with Experiment 06's carry-marginal matched design ($S_{\text{iso}}$ vs $S_{\text{comp}}$, 50% ones / 50% tens carry invariant) tests whether prequential online code length and starting-difficulty probes jointly predict adaptation cost when switching compositional regimes.
- **How to get started on the PR:**
  ```bash
  git fetch origin
  git checkout exp07-prequential-mdl-and-length-transfer
  python3 experiments/07_prequential_mdl_and_length_transfer/expt_v1/test_laws.py
  python3 experiments/07_prequential_mdl_and_length_transfer/expt_v1/run.py --check-only
  ```
- **What is left to close the PR:**
  1. Run the 200 trajectories (`run.py --workers 8`), run `analyze.py`, `verify.py`, and `report.py`.
  2. Inspect whether prequential code length and `stable_difficulty` pass the frozen gates in `experiments/07_prequential_mdl_and_length_transfer/PROTOCOL.md`.
  3. Archive to `experiments/archive/07_prequential_mdl_and_length_transfer/` and merge the PR!
