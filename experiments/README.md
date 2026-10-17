# Experiment index

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/README.md>

**TLDR:** Six controlled addition studies are complete and archived: exposure reverses difficulty rankings after exact answer matching and per-column carry-marginal matching (cutting input-digit TV distance by $>11\times$ to `0.0398`), constrained monotone ridge repairs extrapolation blow-up across held-out widths (up to 384), probed starting difficulty cuts held-out post-training RMSE by **47.20% [42.88, 53.08]** when starting checkpoints have heterogeneous pretraining histories, and balanced post-training recovers **87.55% [83.60, 90.72]** of a pretraining carry gap while passing the strict no-harm retention gate (**+1.09 [+0.33, +1.89] pp** on control). All outcomes and reproduction artifacts are retained.

| Experiment | Setup | Status |
|---|---|---|
| [experiments/archive/01_fixed_width_addition_exposure/](archive/01_fixed_width_addition_exposure/) | Fixed three-digit addition; eight paired seeds, four training distributions | COMPLETE; 94.34% gap recovery, strict retention criterion not established (-6.08 pp) |
| [experiments/archive/02_heldout_exposure_difficulty_laws/](archive/02_heldout_exposure_difficulty_laws/) | Fixed addition; held-out widths, mixtures and questions; 200 trajectories | COMPLETE; primary criterion not established (6.80% combined; 15.80% difficulty-only ablation) |
| [experiments/archive/03_answer_matched_alignment/](archive/03_answer_matched_alignment/) | Answer-matched no-carry versus two-carry addition; rank reversal and positive predictive laws | COMPLETE; ranking reversal supported (25.30 / 24.58 pp), unconstrained difficulty law blew up |
| [experiments/archive/04_monotone_ridge_heldout_addition/](archive/04_monotone_ridge_heldout_addition/) | Fresh models/questions/mixtures (width 384 holdout); constrained ridge versus old predictive laws | COMPLETE; extrapolation repair (100.00%) and ranking reversal supported; shared-warmup feature value 3.48% [1.12, 6.24] |
| [experiments/archive/05_pretraining_history_and_retention/](archive/05_pretraining_history_and_retention/) | Heterogeneous pretraining histories ($q_{\text{pre}} \in \{.1,.2,.5,.8,.9\}$) followed by balanced post-training ($q_{\text{post}}=.5$) | COMPLETE; **ALL GATES SUPPORTED** — starting-difficulty RMSE reduction **47.20% [42.88, 53.08]**, gap recovery **87.55%**, no-harm retention **+1.09 [+0.33, +1.89] pp** |
| [experiments/archive/06_compositional_carry_marginal_matched/](archive/06_compositional_carry_marginal_matched/) | Exact answer + 50%/50% per-column carry-marginal matched isolated single-carry vs cascading two-carry addition (input-digit TV `0.0398`) | COMPLETE; **pure compositional reversal SUPPORTED** (**22.35 [20.60, 24.00] pp** and **17.74 [14.62, 21.12] pp**); shared-warmup feature value 1.94% |

[Archive index](archive/README.md) · [Collaborator onboarding & next PRs](ideas/COLLABORATION_ONBOARDING.md).
