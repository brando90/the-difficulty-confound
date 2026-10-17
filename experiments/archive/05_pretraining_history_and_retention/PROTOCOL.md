# Historical exposure, starting-difficulty laws, and no-harm retention

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/05_pretraining_history_and_retention/PROTOCOL.md>

**TLDR:** Test whether independently probed starting difficulty ($D_{m_0}$) substantially improves held-out post-training forecasts under the frozen constrained monotone ridge law when starting checkpoints have heterogeneous prior training histories ($q_{\text{pre}}$), and whether balanced post-training remediation recovers most of the initial carry gap without exceeding the 5-percentage-point no-carry retention drop limit.

Frozen on 10-16-2026 before training.

## Fixed skill and two-stage training design

Reuse the byte-identical `addition.py` reference, six one-hot input digits, four output digits, two-hidden-layer network, carry definition, AdamW optimizer (`lr=1e-3`, `weight_decay=1e-4`), and batch size (`256`) from Experiments 02–04. Use the same 117 eligible answers (<1000) with at least 60 unordered pairs in both no-carry ($c=0$) and two-carry ($c=2$) strata; 40 training pairs per answer (4,680 pairs per mixture).

- **Fresh seeds:** `4101, 4202, 4303, 4404, 4505, 4606, 4707, 4808`.
- **Fresh evaluation splits (`split_seed=20261025`):** 8 test, 4 calibration, and 4 probe pairs per answer and carry stratum (1,872 test, 936 calibration, 936 probe items), excluding all evaluation items (`test`, `calibration`, `probe`) from both Experiment 03 and Experiment 04.
- **Stage 1 (Pretraining History, 1,500 updates):** Each starting checkpoint $m_0(s, w, q_{\text{pre}})$ is pretrained for 1,500 updates across fit widths `32/64/128`, held-out widths `96/384`, fit pretraining mixtures `0.1/0.5/0.9`, and held-out pretraining mixtures `0.2/0.8` (200 starting checkpoints).
- **Independent Starting-Difficulty Probe:** Before Stage 2 begins, evaluate each starting checkpoint $m_0$ on the 936 independent probe items to measure learner-relative starting difficulty $D_{m_0}$ (`starting_bits`). Also evaluate pre-intervention (`step=0`) calibration and test splits.
- **Stage 2 (Balanced Post-Training Remediation, $q_{\text{post}}=0.50$, checkpoints `100/300/1000`):** From each starting checkpoint $m_0$, run balanced post-training ($q_{\text{post}}=0.50$) for 1,000 updates, evaluating calibration and test splits at 100, 300, and 1,000 updates (200 post-training trajectories).

## Six frozen predictive laws

Fit only on the 54 allowed calibration cells per seed (`widths in {32, 64, 128}` and `q_pre in {0.1, 0.5, 0.9}` across 3 checkpoints and 2 carry strata). Freeze all 4,608 held-out predictions before opening final-test outcomes. Retain the exact penalty ($\lambda=0.01$) and coefficient bounds ($\beta_N, \beta_T \le 0$, $\beta_D \in [0, 1]$) from Experiment 04:

1. `legacy_base`: OLS with `intercept`, `log_parameters`, `log_updates`, `two_carry`.
2. `legacy_difficulty`: OLS adding `log_starting_bits`.
3. `ridge_difficulty`: Ridge ($\lambda=0.01$) adding `log_starting_bits` without sign/elasticity bounds.
4. `stable_base`: Monotone ridge ($\beta_N, \beta_T \le 0$) without starting difficulty or hidden pretraining history.
5. `stable_difficulty`: Monotone ridge ($\beta_N, \beta_T \le 0$, $\beta_D \in [0, 1]$) adding `log_starting_bits` from the independent pre-intervention probe.
6. `stable_oracle_history`: Monotone ridge ($\beta_N, \beta_T, \beta_{q_{\text{pre}}} \le 0$) replacing `log_starting_bits` with the true pretraining history exposure $\log q_{\text{pre}, c}$.

## Prospective decision gates

1. **Primary starting-difficulty feature-value gate (`stable_difficulty` vs `stable_base`):** At least 10% relative joint held-out RMSE reduction (192 targets across widths `96/384` and histories `0.2/0.8`), positive 95% paired seed-bootstrap lower bound, positive improvement at both held-out widths (`96` and `384`), and zero invalid forecasts.
2. **Primary targeted remediation & strict no-harm retention gate ($q_{\text{pre}}=0.1 \to q_{\text{post}}=0.5$ at 1,000 updates):** Initial no-carry minus two-carry gap 95% lower bound $\ge 20$ percentage points; fraction of initial two-carry gap recovered 95% lower bound $\ge 50\%$; post-remediation two-carry accuracy $\ge 90\%$; and no-carry control accuracy drop at most 5.0 percentage points (estimate and 95% lower bound $\ge -5.00$ percentage points).
3. **Secondary pre-intervention ranking-reversal gate (`step=0`, $q_{\text{pre}}=0.1$ vs $0.9$):** Both endpoint advantages $\ge 10$ percentage points, positive 95% lower bounds, and positive at all five widths.
4. **Separate extrapolation-repair gate (`stable_difficulty` vs `legacy_difficulty`):** Identical 10% and positive-bound criteria on joint holdout.
