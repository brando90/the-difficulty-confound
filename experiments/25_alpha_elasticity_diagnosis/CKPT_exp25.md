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
