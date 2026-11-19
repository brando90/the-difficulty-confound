# CKPT: Experiment 25, why disjoint distractors give $e_\alpha \approx 0.5$

Dated log (US Pacific). Owner: Laolu Oguneye. Branch `exp25-alpha-elasticity-diagnosis` (cut from `exp24-similarity-alignment`, because it reuses Experiment 24's and Experiment 13-GPU's code). Laptop environment: Python 3.12, torch 2.14.1 (CPU), numpy 2.5.4, scipy 1.18.1. SNAP environment: skampere2 `/lfs/skampere2/0/laoluog1/venvs/tdc_exp13` (torch 2.6.0+cu126), the one Experiments 13-GPU and 24 ran in.

## 11-08-2026

- About 01:45 Laolu chose Experiment 24's open question as the next experiment, over GPU plan g3.
- About 01:50 Exploratory look at Experiment 24's saved results (`tstar.csv`, `metrics.csv`), disclosed in PROTOCOL Section 1.2:
  - local α slopes are steeper at low α;
  - at n = 8, disjoint α = 0.2 is cheaper in total tokens than α = 1 (S 6 of 9, M 7 of 9);
  - α = 1 runs start badly at Experiment 24's 4× learning rate (M, n = 8: 7.5 bits at $2^{20}$ vs 2.8 at α = 0.2);
  - curves flatten late at low α.

  These shaped hypotheses H1–H4.
- About 01:53 Laolu approved writing up the recommended design (disjoint pools only; n = 8 and 32; α 0.02–1; arms `hmm` / `iid` / `masked`; learning rate 0.5×–8× with a per-cell tuned policy; S and M; 3 new seeds; 2,880 runs). The full design is reviewed before the freeze commit.
- 01:54–02:02 Code:
  - `concepts25.py`, `run25.py`, `analyze25.py`, `verify25.py`, `test_exp25.py`, `launch/autopilot25.sh`, `launch/setup25.sh`.
  - Experiment 24's `concepts24.py` and `analyze24.py` and Experiment 13's `hmm.py`, `model.py`, `run.py`, `gpu_arm.py` are imported read-only.
  - Unit tests: 8 of 8 pass.
- 02:02–02:04 `run25.py --check-only` on the laptop: PASS in 134 s.
  - The 30 kept targets and disjoint distractors are identical to Experiment 24's committed `tasks.json` (definitions and concept ids).
  - The 24 iid sources match their parents' document unigram (total variation ≤ 0.011, support 27–95 tokens). Entropy: 4.91 bits/token vs the parents' 2.88, disclosed in PROTOCOL.
  - The 60 α schedules are exact over 2,097,152 documents, and every batch has at least 2 target documents.
  - 0 of 9,216 evaluation documents appear among 10,616,832 training documents.
  - 2,880 shards; smoke shards 0 and 2817.
  - ε_id is identical for both pools (0.044 bits at α = 0.02).

  Receipt: `expt_v1/results_check.json`.
- 02:06 Synthetic full-grid analysis (2,880 planted runs, 400 bootstrap draws, 4 s): planted slopes recovered (above α = 0.2: 0.56 vs 0.55 planted; below it: 0.77 vs 0.80); the independent fitter agrees; diverged runs are handled as censored. At this speed 20,000 draws take about 3–4 minutes.
- Unit tests before the freeze: Experiment 25 8 of 8.
- **Freeze commit, "Freeze Experiment 25 protocol"** (the commit that adds this line), approved by Laolu after reviewing the design. No run of this experiment exists before it. From here on PROTOCOL Sections 1–7, `config.json` and the code are read-only.
- 03:16 First launch on skampere2 (`setup25.sh c0db133d`). 03:17 the autopilot halted at stage 0: `test_gates_on_planted_data` failed (1 failed, 7 passed). The independent check's refit disagreed by a relative 1.16 × 10⁻⁴ (tolerance 10⁻⁴), because scipy 1.16.3's default L-BFGS-B stopping rule can stop early. No prep, smoke or shard ran.
- About 14:45 **Pre-run amendment** (PROTOCOL Section 8): the independent L-BFGS-B refit now runs to a tight tolerance (ftol 10⁻¹⁵, gtol 10⁻¹⁰). Worst disagreement on the planted tables: 4.4 × 10⁻⁶ and 2.4 × 10⁻⁵. Unit tests 8 of 8 on the laptop. Only `independent_check` in `analyze25.py` changes. Approved by Laolu. The commit that adds this line is the new freeze.
- 14:45 Relaunch on skampere2 at `d6e6114c` (the halted launch's logs are in `runs/logs/first_launch_c0db133d/`). 14:47 unit tests pass (8 of 8). 14:52 prep (check-only) PASS. 14:54 GPU smoke: shards 0 and 2817 trained and VERIFIED. 14:55 the queue of 2,880 shards starts, M first.

## 11-11-2026

- 10:40 Shards finished: 2,880 of 2,880 VERIFIED. 10:41 merged into `expt_v1/results/`. The autopilot waits for `state/PREDICTIONS_SHA` before the analysis (PROTOCOL Section 5). No analysis has run, and no result has been looked at.

## 11-19-2026

- **Laolu's predictions** (PROTOCOL Section 5), recorded before any analysis. Gates use the tuned learning rate, pooled over S and M (PROTOCOL Section 4.4). For reference, Experiment 24's disjoint $e_\alpha$ at the fixed 4× rate was 0.52.

  | Gate | Claim | Predicted verdict |
  |---|---|---|
  | G1 | $e_\alpha$(hmm, tuned) − $e_\alpha$(hmm, 4×) > 0: one rate for all α biases the slope down | NOT_ESTABLISHED |
  | G2 | $e_\alpha$(masked) < 1: target-only learning is not purely token-limited (opposite: "> 1") | SUPPORTED |
  | G3 | $e_\alpha$(masked) − $e_\alpha$(hmm) > 0: disjoint HMMs substitute for target data (opposite: "interference") | SUPPORTED |
  | G4 | $e_\alpha$(iid) − $e_\alpha$(hmm) > 0: the substitution needs sequential structure | SUPPORTED |
  | G5 | $e_\alpha^{\text{low}}$ − $e_\alpha^{\text{high}}$ > 0 (hmm, hinge at α = 0.2): part of the bill is share-independent | SUPPORTED |

  | Arm (tuned learning rate) | Predicted $e_\alpha$ |
  |---|---|
  | hmm (disjoint HMM distractors) | 0.5 |
  | iid (unigram-only distractors) | 0.65 |
  | masked (distractors give no gradient) | 0.8 |

  Laolu made these picks after a briefing from the assistant that explained each gate and gave prior ranges for the three slopes from Experiment 24's results and the Section 1.2 clues (hmm about 0.5–0.75, masked about 0.5–0.9, iid about hmm to hmm + 0.15). Neither had seen any Experiment 25 analysis or result. Laolu first wrote "above 1" for G2, then chose G2 SUPPORTED (masked < 1) to match the masked slope of 0.8.
- 13:52 The autopilot read `state/PREDICTIONS_SHA` (9e4912b0) and started the analysis. 14:02 ALL_DONE: analysis (20,000 bootstrap draws, independent fitter AGREE) and merged `verify25.py` PASS in 434 s. Verdicts: G1 NOT_ESTABLISHED; G2, G3, G4, G5 SUPPORTED, matching all five predictions. Measured e_α (tuned): hmm 0.365, iid 0.453, masked 0.438, all below the predicted 0.5, 0.65 and 0.8.
- Results copied from skampere2 (SHA-256 of every file identical to the copy there). `report25.py` writes `RESULTS_alpha_elasticity.md` and the figure. Disclosed in the results: 33 of 64 tuned learning-rate choices are at the edge of the 0.5×–8× grid (α = 1 chose 8× in 3 of 4 cells), and the T0 + T1/α fit is exploratory, not part of the frozen analysis. No runtime limit of PROTOCOL Section 7 was raised during the run. Kept checkpoints: 155 GB on `/dfs/scratch1/laoluog1/tdc_exp25/runs/shards`.
