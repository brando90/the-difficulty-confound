# Historical exposure and retention results

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/05_pretraining_history_and_retention/results.md>

Status: DONE — 200/200 post-training trajectories and 200/200 pretraining checkpoints complete; deterministic verification passed.

**TLDR:** When starting checkpoints have heterogeneous pretraining exposure histories, adding independently probed starting difficulty (`stable_difficulty`) reduces joint held-out forecast error by **47.20% [42.88, 53.08]** over `stable_base` (**SUPPORTED**), outperforming even the oracle pretraining-history law (`stable_oracle_history`, 0.1580 vs 0.1004 bits/answer). Balanced post-training remediation recovers **87.55% [83.60, 90.72]** of the initial two-carry gap while improving control no-carry accuracy by **+1.09 [+0.33, +1.89]** percentage points, passing the strict no-harm retention criterion (**SUPPORTED**) that Experiment 01 missed. [Full results and uncertainty](RESULTS_history_retention.md).

| Frozen comparison | Measured result | Verdict |
|---|---|---|
| Starting-difficulty feature value (`stable_base` → `stable_difficulty`) | 0.1902 → 0.1004 bits/answer; reduction 47.20% [42.88, 53.08] | Supported |
| Oracle pretraining-history benchmark (`stable_base` → `stable_oracle_history`) | 0.1902 → 0.1580 bits/answer; reduction 16.94% [13.40, 24.82] | Supported |
| Targeted gap recovery & strict no-harm retention ($q_{\text{pre}}=0.1 \to q_{\text{post}}=0.5$) | 87.55% [83.60, 90.72] gap recovered; control change +1.09 [+0.33, +1.89] pp (gate ≥ -5.00 pp) | Supported |
| Pre-intervention difficulty ranking reversal (`e_low`, `e_high` at `step=0`) | 39.01 [36.31, 41.57] and 39.27 [37.02, 41.54] percentage points | Supported |

95% paired bootstrap intervals over eight seed blocks (20,000 draws); fixed test set with 936 questions per carry stratum; joint prediction comparison has 192 targets per law. p-val=n/a.

Completed in 1.67 minutes on skampere2 using eight parallel seed workers (two CPU threads per worker); frozen source `13a0bf6a4e0f5d2320b2ae29ee60faf27deef3c4`. All 200 final weights and 200 warm weights replayed with 0.0 bits discrepancy; 3,200 aggregate rows reconciled from 2,246,400 item observations; all 4,608 forecasts and active-set optima independently reconstructed. Canonical Mac verified all 2,231 artifact SHA-256 hashes. [Full report](RESULTS_history_retention.md) · [Protocol](PROTOCOL.md) · [Checkpoint](CKPT_history.md).
