# Results: Experiment 19 diagnostic, cosine learning-rate decay at alpha = 1 (AT-494)

Run on 10-19-2026 on one Sherlock GPU (NVIDIA L40S, node sh04-01n09), job 48512967, 6:29 of wall time (389 s), **0.108 GPU-hours** (sacct ElapsedRaw x 1 GPU; `results/sacct.txt`). Pre-registered in PREREG.md at commit 3f0b6670 before any GPU run; nothing in PREREG Sections 2-5 changed. Both arms reached 2^20 (no control drop, stop reason "reached T_max", 360 s of training and evaluation).

## Verdict

**Retire the transformer arm; the LSTM arm carries the DFA question.** The decision rule needed the cosine arm's median below 0.5 bits at both n = 16 and n = 32 at 2^20. It got **0.851 bits at n = 16 and 1.127 at n = 32**. Cosine decay did not help at all: it was slightly *worse* than the same-stream constant control at n = 16 (14 of 16 paired cells, sign test p = 0.004) and n = 32 (13 of 16, p = 0.021). So the n = 16/32 stall in Experiment 19 is not a constant-learning-rate optimization failure, and rerunning the grid with this schedule would not buy convergence.

## Numbers at T = 2^20 (16 cells per level: 4 seeds x 4 targets)

Median excess bits on Experiment 11's held-out test set (cells below 0.5 bits in brackets); paired = cosine minus constant on the same cell.

| n | constant | cosine | paired median | cosine lower | sign test p |
|---|---|---|---|---|---|
| 4 | 0.040 (14/16) | 0.021 (13/16) | -0.014 | 11/16 | 0.210 |
| 8 | 0.366 (10/16) | 0.405 (9/16) | +0.057 | 5/16 | 0.210 |
| 16 | 0.829 (2/16) | 0.851 (1/16) | +0.058 | 2/16 | 0.004 |
| 32 | 1.054 (0/16) | 1.127 (0/16) | +0.059 | 3/16 | 0.021 |

Median curves at 2^17 / 2^18 / 2^19 / 2^20:

| n | constant | cosine |
|---|---|---|
| 16 | 0.860 / 0.803 / 0.796 / 0.829 | 0.857 / 0.797 / 0.777 / 0.851 |
| 32 | 0.974 / 0.959 / 0.983 / 1.054 | 0.975 / 0.966 / 0.988 / 1.127 |

## Prediction versus outcome

| Prediction (PREREG Section 3) | Outcome |
|---|---|
| Cosine lowers n = 16 relative to constant in most paired cells | Wrong: cosine was lower in only 2 of 16 (median +0.058 bits) |
| n = 16 cosine median about 0.55-0.75, not below 0.5 | Not below 0.5, as predicted, but higher than the range: 0.851 |
| n = 32 stays near chance (median > 0.9) | Right: 1.127 |
| Verdict: retire | Right |

## What else it shows

- **The stall reproduces on a fresh data stream.** The constant arm (Experiment 19's schedule on stream seed 494019) ended at 0.829 (n = 16) and 1.054 (n = 32) bits, against Experiment 19's 0.849 and 1.036 for the same cells. So the stall is a property of this learner, not of Experiment 19's particular data.
- **Loss still rose in the last doubling while the learning rate went to zero.** Cosine n = 16 went 0.777 to 0.851, n = 32 0.988 to 1.127, from 2^19 to 2^20, more than the constant arm rose. Step-size noise cannot explain a rise under a vanishing step size; it looks more like the model settling into a solution that generalizes worse to the held-out strings (for example, fitting the training partition). This run does not test that, and it does not change the verdict.
- n = 4 and n = 8 behave as in Experiment 19: n = 4 learns (0.02-0.04 bits), n = 8 is partway (0.37-0.41 bits), with no significant difference between the arms.

## Verification

`diag_verify.py --results results` printed **PASS** on a second machine (not Sherlock) from the fetched archive (`results/verification.json`): sources match the manifest; all 1536 metrics rows recomputed from saved predictions and equal to the CSV, every lr on schedule; all 128 final checkpoints (64 cells x 2 arms) replayed on CPU in Experiment 11's model.py, max mean-bits difference 1.2e-7; test set identical to Experiment 11's and init hashes match; diag_summary.py rerun gives an identical summary.json. Archive sha256 66be5bc47d3cc94057ad80b8e77b1beb6b9dfda1b210ee9c0c22e866f29a882a. Per-string predictions and checkpoints stay out of git (they are in the archive on Sherlock, $SCRATCH/tdc_gpu/diag494_results.tgz).

## Next step

Move the DFA question to the LSTM arm (it learned n = 4-16 on CPU in Experiment 11) and stop spending GPU hours on this transformer.
