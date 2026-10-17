# Experiment 05 checkpoint
Created: 10-16-2026 17:05 PDT
Last updated: 10-16-2026 17:12 PDT

## Goal
Test starting-difficulty predictive value under heterogeneous pretraining histories and strict no-harm retention under balanced post-training remediation.
## What ran
Frozen source `13a0bf6a4e0f5d2320b2ae29ee60faf27deef3c4`; all preflight checks passed; 200/200 pretraining checkpoints and 200/200 post-training trajectories completed in 1.67 minutes on skampere2; all 2,231 artifacts transferred and verified on canonical Mac.
## Results
- Starting-difficulty feature value (`stable_base` → `stable_difficulty`): **SUPPORTED** (0.1902 → 0.1004 bits/answer; 47.20% [42.88, 53.08] reduction; 0 invalid forecasts).
- Targeted gap recovery & strict no-harm retention ($q_{\text{pre}}=0.1 \to q_{\text{post}}=0.5$): **SUPPORTED** (87.55% [83.60, 90.72] gap recovered; target accuracy 56.54% → 90.70%; control accuracy change +1.09 [+0.33, +1.89] pp, passing the ≥ -5.00 pp gate).
- Pre-intervention ranking reversal (`step=0`): **SUPPORTED** (`e_low` 39.01 [36.31, 41.57] pp, `e_high` 39.27 [37.02, 41.54] pp).
## Stuck / blocked
None.
## Next steps
Archive under `experiments/archive/05_pretraining_history_and_retention/` and proceed to Experiment 06 (`06_compositional_carry_marginal_matched`).
