# CKPT: Experiment 11, exact DFA complexity versus data alignment

Dated log of every step of the Experiment 11 run on the CPU box (4 logical cores, 15 GB RAM, no GPU), branch `results/cpu-expts-11-15`. Times are US Pacific. Commands are run from `experiments/11_dfa_exact_complexity_vs_alignment/expt_v1/` with the pinned environment of requirements.txt (Python 3.12.3, torch 2.14.1+cpu).

## 10-18-2026

- 11:08 Working clone at snapshot commit 45302443 (draft designs for Experiments 11-16).
- 11:28 Code and unit tests committed (b9ec88f1): dfa.py, data.py, model.py, run.py, test_dfa.py, test_data.py, test_model.py. Deviation: test_model.py compares batched evaluation with a naive per-string loop at tolerance 1e-5 instead of 1e-12. The measured maximum difference is 7.4e-7, from float32 kernels.
- 11:28 `run.py --check-only`, receipt in results_check.json (937ea904). All checks pass: Hopcroft equals Moore for all 32 tasks; tasks pairwise distinct and not complements; test set 4,096 strings (241 per length, 240 at length 24); 0 test strings in pools; schedule within 4 examples of αT; no pool reuse in the gated grid; parameters 104,002 (transformer) and 84,290 (LSTM). The first timing probe ran while the box was busy and projected 5.16 h. A rerun projected 2.46 h (69.1 s per cell), and the receipt keeps the rerun.
- 11:28 Pilot launch 1 (`run.py --pilot`, seed 11909) died while building the data, when its launching command exited. No training had happened. results_pilot/ was removed and the pilot relaunched as a background job at 11:28:49.
- 11:45 Pilot finished (about 16 minutes). At α = 1 and T = 2^16 the losses were 0.31 to 0.99 bits (median 0.90). No target reached 0.1 bits even at 2^17. The n = 16 and n = 32 targets stayed at 0.8 bits or more. 15 of 16 targets were still improving from 2^16 to 2^17, 7 of them by more than 10%. P1 fires (median T*(0.1) censored at 2^17). P2 does not fire (0 of 16 targets at or below 0.1 bits at 2^10). P3 does not fire (projected 15,178 s against 16,200 s).
- 11:50 Pre-registered before any calibration result was read (the protocol was still a draft): a learning-rate calibration on fresh seed 11910 (never reused), α = 1, all 16 targets, window 2^9 to 2^16, learning rates 3e-3 and 1e-2. Learning rate 1e-3 is read from the pilot at 2^16. Rule: lowest median over the 16 targets of ε(2^16), with non-finite losses counted as infinite. P1 and P2 are then re-applied to the chosen learning rate. New rule P1b: if the median T*(0.3 bits) is also above 2^14, the primary ε* becomes 0.5 bits and the levels become {0.5, 0.3, 0.1}.
- 11:50-12:02 Calibration ran (`run.py --calibrate-lr 0.003`, then `0.01`, about 11.5 minutes in total). Medians of ε(2^16): 0.899 bits (1e-3), 0.931 (3e-3), 0.977 (1e-2). The learning rate stays 1e-3. P1 fires again, and P1b fires: the primary ε* is 0.5 bits, the levels are {0.5, 0.3, 0.1}, and the G4 model-free fallback is ε* = 0.7 bits.
- 12:03 Analysis code, law tests, pilot and calibration committed (e60ba9c9). The pilot disclosure was written into PROTOCOL Section 2.6, and config.json was updated for P1b.
- 12:03 **Freeze commit 4caa44f1, "Freeze Experiment 11 protocol".** From here on, Sections 2-5 of PROTOCOL.md and every gate value in config.json are read-only.
- 12:03 Gated grid chunk 1 launched: `run.py --output results --max-seconds 6600` (4 workers, 1 thread each, longest cells first).
- 13:15 A synthetic dry run of analyze.py and report.py (fake metrics in a scratch directory, no gated result read) found a bug. Stage 2 raised an error because the E_mf fallback ε* (0.7 bits) was not in the T* table. The fix computes that T* on the fly with the same t_star function, committed as e3e15c43. No gate, threshold, split, seed or ε* changed.
- 14:05 Chunk 1 stopped scheduling at its time budget: 382 of 512 cells in 6,619 s. Chunk 2 launched with `--resume`, the only resume. Deviation: the protocol's single run became two resume-safe chunks because a single foreground command is limited to 60 minutes. Both chunks are logged in manifest.json. Done markers are written last, so the second chunk skipped completed cells only, and no cell was rerun.
- 14:46 Chunk 2 finished: 130 cells in 2,456 s. **Grid complete: 512 of 512 cells, 0 missing, 9,075 s cumulative** (cap 18,000 s).
- 14:48 Optional LSTM arm launched: `run.py --lstm --output results --resume` (α = 1, 16 targets, 8 seeds; outputs in results/lstm_arm/).
- 14:59 `analyze.py --stage 1`: G2 K law chosen on fit rows only, M_K_exp (fit RMSE 0.1377 against 0.1409 for M_K_coef). laws.py and independent_fit.py agree. **forecasts_frozen.json SHA-256 `ea5bd67245fb807086b5ac8d7f29b566af7e74d9e6c55f9b7f7f72ba297d1b53`.** Held-out rows were read only after this line.
- 15:00 `analyze.py --stage 2`: G1, G2, G3a, G3b, G4, G5, G6 and G7 are all NOT_ESTABLISHED, and C3 is unresolved. Key numbers are in RESULTS_dfa_complexity_alignment.md.
- 15:02 `verify.py` started while the LSTM arm was still running (44 of 128 cells).
- 15:41 LSTM arm finished: 128 of 128 cells in 3,185 s.
- 16:05 `verify.py` finished: **PASS** in 3,738 s (about 62 minutes, sharing the CPU with the LSTM arm until 15:41). Training sources match manifest.json. 32 of 32 tasks regenerate identically, and Hopcroft and Moore give the same n for all of them. Covariates are exact and agree with Monte Carlo for 32 of 32. Data regenerate identically for 32 of 32 and mixes for 512 of 512. 4,096 checkpoints were replayed with a maximum difference of 0 in excess bits, accuracy and per-string predictions. T* is identical, the forecast hash is identical, the two fitters agree, and the gates and key numbers are identical. No verifier fixes were needed.
- 16:06 Unit tests rerun after the grid: 35 passed (689 s).
- 16:17 `report.py --notes ... --tldr ...` wrote RESULTS_dfa_complexity_alignment.md and dfa_complexity_alignment_results.png/.pdf.
- 16:18 `tar -cf results/checkpoints_dfa.tar -C results checkpoints`: 1,740,902,400 bytes, SHA-256 `d160e8bf61da7c8b0ae3a0184c27d7ce7d323cbe2d137c7821b815355dd4fa77`. ARTIFACT_SHA256.json records the SHA-256 of all 5,583 files under expt_v1/results/, including the tar and every checkpoint. The checkpoints and the tar stay out of git.
- 16:19 Results committed locally on `results/cpu-expts-11-15` (not pushed). The archive move, README rows, push and pull request are left to the parent task.

### Wall time

Pilot about 16 minutes, calibration about 11.5 minutes, gated grid 9,075 s (2 h 31 min), LSTM arm 3,185 s, verification 3,738 s, unit-test rerun 689 s. Peak resident memory of verify.py was 0.3 GB, and the artifacts are 1.74 GB, under the 2 GB cap.

### Deviations (also in RESULTS)

1. The grid ran as two resume-safe chunks (one resume).
2. A learning-rate calibration and rule P1b were added before the freeze and disclosed.
3. The code used an existing environment with the pinned versions instead of ~/venvs/exp11.
4. The test_model.py tolerance is 1e-5 instead of 1e-12.
5. One analyze.py bug fix after the freeze, made before any gated result was read.
6. verify.py check 1 gates on the training sources, and the analysis-source hashes are recorded.
7. No archive move, push or pull request in this run.
8. The E table uses the frozen levels {0.5, 0.3, 0.1}.

The pilot reused pool strings between 2^16 and 2^17 (pilot only).

**TLDR-end:** [tdc: Expt 11 CKPT] Pilot, pre-registered learning-rate calibration and freeze (4caa44f1), then 512 of 512 gated CPU runs in 9,075 s with one resume, frozen forecasts hashed before held-out rows were read, all gates NOT_ESTABLISHED because the transformer barely learned within 2^16 examples, an exploratory LSTM arm that did learn, and replay verification PASS.
