# CKPT: Experiment 24, similarity of the non-target data × complexity × alignment

Dated log (US Pacific). Owner: Laolu Oguneye. Branch `exp24-similarity-alignment` (stacked on `exp13-ginc-gpu`, PR #28, because it reuses Experiment 13-GPU's code). Laptop environment: Python 3.12, torch 2.14.1 (CPU), numpy 2.5.4, scipy 1.18.1. SNAP environment: skampere2 `/lfs/skampere2/0/laoluog1/venvs/tdc_exp13` (torch 2.6.0+cu126, numpy 2.2.6, scipy 1.16.3), the one 13-GPU ran in.

## 11-03-2026

- 02:40 Design decisions made (Laolu signed off on the recommended choices, plus: α = 0.5 added; complexity levels n = 8, 32, 64; sizes S and M; ε* = 0.1 with a 2^28-token cap; Task2Vec alongside, never a gate; the fitted law secondary; only cost cuts of negligible effect). Decided here, with reasons in PROTOCOL Sections 2 and 7:
  - every run trains to the 2^28 cap (no early stopping);
  - a real learning-rate pilot at 1×/2×/4× the 13-GPU table rate (13-GPU's choice was the top of its grid);
  - 3 targets per level;
  - 4 distractors per pool;
  - a 301-token vocabulary for every run;
  - a two-stage (seed, target) bootstrap plus the every-seed rule.

  Cost cuts adopted: 1,024 evaluation documents, and deleting the 4 odd-power checkpoints after their replay. Laolu's constraint: skampere1's runs take precedence (skampere1 is not used), and on skampere2 a job starts only on a GPU with ≥ 24 GB free and ≤ 30% utilization.
- 02:45 Node state: skampere2's 8 H200s all at 74–100% utilization with 3.5–13 GB free (Laolu's distillation runs and others), so the queue waits until GPUs free up.
- 02:50–03:30 Code: `concepts24.py`, `run24.py`, `analyze24.py`, `verify24.py`, `task2vec24.py`, `test_exp24.py`, `launch/autopilot24.sh`, `launch/setup24.sh`. Experiment 13's `hmm.py`, `model.py`, `run.py`, `gpu_arm.py` are imported read-only. Unit tests: 10 of 10 pass. One test was redesigned: a planted "no effect" case could pass G5 by noise, so the test now plants a clear effect and a clear null.
- 02:55 Local concept build: 160 concepts (9 targets, 3 pilot concepts, 148 distractors) in 162 s, h_ref 2.863, band ±0.15 (no widening). Shared fractions: copy20 0.75–0.81, copy50 0.50, family and disjoint 0. Vocabulary overlap with the target: copy20 0.64–0.88, copy50 0.42–0.80, family 0.10 (n = 8) to 0.71 (n = 64), disjoint 0. ε_id = log2(1/α)/128 at every level (0.052 bits at α = 0.01). Posterior on the target after 16 tokens at α = 0.05: copy20 0.87, copy50 0.99, family and disjoint 1.00.
- 03:20 Synthetic end-to-end analysis (918 planted runs, 300 bootstrap draws, 16 s): planted e_α 0.10 / 0.25 / 0.40 / 0.80 recovered as 0.099 / 0.250 / 0.400 / 0.802; independent Tobit fitter agrees to 3.7e-6. It exposed one division by zero when every held-out run is censored (G3 now reads NaN → NOT_ESTABLISHED). Task2Vec smoke (CPU, 2^18-token probe) runs at about 27 s per target.
- 03:05 `run24.py --check-only` on the laptop: PASS. 160 concepts certified (rank = n, K_bits recomputed); 144 α schedules exact over 2,097,152 documents (distractor spread ≤ 1); 0 of 10,336 evaluation documents among 41,943,040 training documents (first 2^16 of every concept under every seed and the pilot seed); 918 shards; parameters S 3,269,376 / 3,159,552 and M 25,439,744 / 25,220,096. Receipt: `expt_v1/results_check.json`.
- Unit tests before the freeze: Experiment 24 10 of 10, Experiment 13 30 of 30.
- **Freeze commit, "Freeze Experiment 24 protocol"** (the commit that adds this line). No run of this experiment exists before it. From here on PROTOCOL Sections 1–7, `config.json` (except `lr_selected`, by the pilot rule) and the code are read-only.
