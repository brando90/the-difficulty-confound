# Experiment 19 diagnostic: is the n=16/32 stall a constant-learning-rate failure? (AT-494)

Status: pre-registered on 10-19-2026, committed and pushed to main before any run of this code on a GPU. Seen before writing this: Experiment 19's full results (RESULTS_dfa_transformer_gpu.md, metrics.csv). A CPU smoke test to 2^11 (32 steps) checked the pipeline and verification only; its numbers say nothing about the question. After this commit, Sections 2-5 do not change; deviations go in RESULTS.

## 1. Question

In Experiment 19, at alpha = 1, the n = 16 and n = 32 automata never got below 0.5 bits by 2^20 examples. Seeds 11101-11404, T = 2^20: n = 16 median 0.849 bits (2 of 16 cells below 0.5), n = 32 median 1.036 (0 of 16). Both levels went up in the last doubling (n = 16: 0.788 at 2^19, then 0.849; n = 32: 0.968, then 1.036) at a constant lr of 1e-3. That pattern fits an optimizer that can't settle at that step size. Does decaying the learning rate let these cells learn? If it does, the grid should be rerun with that schedule. If it doesn't, the transformer arm is retired in favour of the LSTM arm.

## 2. Design

- Cells: alpha = 1 only, Experiment 19's first four seeds (11101, 11202, 11303, 11404) x all 16 targets (n = 4, 8, 16, 32, 4 targets each) = 64 cells.
- Initialization: identical to Experiment 19's cells (torch.manual_seed(seed), then Experiment 11's model.build_model). The run records a hash of each seed's init and verification recomputes it.
- Two arms trained side by side on one shared data stream, so the same strings come in the same order for both:
  - `constant`: Experiment 19's schedule, 8 linear warmup steps to 1e-3, then flat.
  - `cosine`: the same warmup, then lr = 1e-3 x 0.5 (1 + cos(pi (s - 8) / (16384 - 8))) for step s >= 8, reaching 0 at 2^20 examples (16384 steps of 64).
- Everything else is Experiment 19's code, unchanged (expt_v1/common.py, ensemble.py and gpu_train.py, imported as is; their sha256 checksums match the Experiment 19 run): 104,002-parameter transformer, AdamW (betas 0.9/0.999, weight decay 0.01, fused on GPU), per-model gradient clip 1.0, batch 64, uniform train-partition strings with lengths 8-24, float32 with TF32 off, and Experiment 11's 4,096-string held-out test set.
- The data stream differs from Experiment 19's (one shared stream, seed 494019). That's why the constant arm is a same-stream control, so Experiment 19's numbers aren't the only comparison.
- Checkpoints at every power of two from 2^9 to 2^20. The final checkpoint is saved for both arms.

## 3. Prediction and decision rule

Primary measure: excess bits on the held-out test set at T = 2^20, the median over the 16 cosine-arm cells at each of n = 16 and n = 32.

Prediction (written before the run): cosine decay will lower n = 16 relative to the constant arm in most paired cells, but not below 0.5 bits (we expect a median around 0.55-0.75). We expect n = 32 to stay near chance (median > 0.9). So we expect the verdict to be retire.

Decision (computed mechanically by diag_summary.py):

- **Rerun the grid with cosine decay** if the cosine arm reached 2^20 and its median is below 0.5 bits at both n = 16 and n = 32. Both are needed because Experiment 11's elasticity fit uses n = 32 (fit_n = 4, 8, 32) and n = 16 is its held-out level.
- **Retire the transformer arm** (use the LSTM arm for this question) otherwise. That includes the case where only n = 16 gets below 0.5.

Reported alongside, with no effect on the decision: cells below 0.5 per level and arm; the paired cosine-minus-constant difference per level, with a two-sided sign test over 16 cells; n = 4 and n = 8; and the median curves. If the constant arm itself gets below 0.5 at n = 16 and n = 32, Experiment 19's stall depended on its data stream. RESULTS then says so, and the verdict still follows the rule above.

## 4. Budget

Brando has 8 GPU-hours on Sherlock in all, and about 1.2 are left (Lean SFT 2.12 + AT-441 3.65 + Experiment 19 1.06). This run targets about 0.3 GPU-h. diag_train.py stops itself at 18 minutes of wall time from process start, and Slurm's limit is 24 minutes (0.4 GPU-h worst case).

- Control-drop rule: decided once, at T = 2^13, from the measured speed of the 2^12 to 2^13 octave. If both arms can't reach 2^20 inside 18 minutes (projected time x 1.05 + remaining evals + 60 s), the constant arm is dropped and only the cosine arm continues, so the cosine schedule always finishes. The comparison then falls back to Experiment 19's constant-lr numbers for the same cells (Section 1).
- Octave rule: before each octave, the run stops cleanly if the next one is projected past the deadline. A cosine arm that stops before 2^20 is a failed diagnostic (verdict retire, flagged as a budget failure, not as evidence).
- GPU hours used come from sacct for the job (ElapsedRaw x GPUs).

## 5. Verification

diag_verify.py must print PASS before any number is reported. It checks:

1. The sources match the run's manifest.
2. Every metrics row is recomputed from the saved per-string predictions, and every lr value matches the schedule formula.
3. Every cell's final checkpoint, from both arms, is replayed on the CPU in Experiment 11's model.py, with mean bits within 1e-4.
4. The test set matches Experiment 11's exactly, and the init hashes match.
5. diag_summary.py rerun in a scratch copy gives an identical summary.json.

Verification runs on a second machine from the fetched archive. That also confirms the result reproduces off Sherlock.

## 6. Files

- `diag_train.py`, `diag_config.json`: the run.
- `diag_summary.py`: the read-out and verdict.
- `diag_verify.py`: verification.
- `diag.sbatch`, `submit_diag.sh`: Sherlock submission.
- `SHA256SUMS`: checksums of the files above.
- Results go in `results/` (predictions and checkpoints stay out of git), and `RESULTS.md` is written after the run.
