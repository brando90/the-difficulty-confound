# CKPT: Experiment 13, GINC-style HMM mixtures, complexity versus alignment

Dated log of the 13-small run on the CPU box (4 logical cores, 15 GB RAM, no GPU), branch `results/cpu-expts-11-15` in the working clone. Times are US Pacific. Commands run from `experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1/` with `~/.venvs/tdc/bin/python` (Python 3.12, torch 2.14.1+cpu). 13-GPU: pending, not run.

## 10-18-2026

- 16:25 PROTOCOL.md and PROMPT.md checked identical to the design task's tree (scratch/at342/repo); no edits needed.
- 16:25-17:03 Code and tests: hmm.py, model.py, run.py, laws.py, independent_fit.py, analyze.py, verify.py, report.py, test_hmm.py, test_model.py, test_laws.py. 21 of 21 unit tests pass.
- 17:03 `run.py --arm small --check-only`, receipt results_check_small.json: 20 concepts rebuilt in prep/small with certified process rank = n, entropy band ±0.15 bits around h_ref 2.847, α schedules exact at every prefix, 0 evaluation/training collisions over 6,553,600 training documents, all oracle and mixture references finite. Observation: ε_id over all 128 positions equals log2(1/α)/128 exactly for every target (distractor documents give the target ~0 probability). Committed 49f84615.
- 17:05 Run stopped for a machine restart; nothing was running.
- 17:10-17:39 Pilot `run.py --arm small --pilot` (seed 13909, 8 runs): 8/8 finished, none diverged, 1,720 s. 3,884 training tokens/s per worker (median), 37.1 s evaluation per checkpoint, with a background load of about 3 from other jobs. Projected 7.50 h at 2^21 and 4.58 h at 2^20, so the fallback T range {2^16..2^20} applies. ε* rule fired: n = 32 pilot at α = 0.4 had ε_learn 1.093 bits at 2^19, so ε* = 1.10. Median evaluation SE 0.0048 bits. Outputs: pilot_summary.json, results_pilot/ (npz copies and checkpoints out of git; npz SHA-256s in results_pilot/npz_SHA256.txt).
- 17:41 Pre-freeze edits: config small.T_checkpoints (fallback), small.epsilon_star 1.1, shared.eps_res 0.015, small.max_wall_seconds 14,400 → 18,000 (deviation, see RESULTS). PROTOCOL Section 2.8 pilot disclosure; Section 5 note aligning G1-G4 with Experiment 11's frozen gates (G5 kept; draft versions reported, not gated). Tests 21/21 pass.
- 17:42 **Freeze commit 19a77724, "Freeze Experiment 13 protocol".** No gated run existed before it.
- 17:42 Grid chunk 1: `run.py --arm small --output results/small --workers 4 --max-seconds 3300`. Killed by the 3,500 s command timeout at 18:40 with 36 runs done and 4 in flight (their `.tmp` checkpoint folders are discarded on rerun). run_state.json had not been written; its chunk-1 entry was reconstructed by hand (3,501 s, 36 runs) and marked as such.
- 18:41-19:31 Chunk 2 (`--max-seconds 2900`): 32 runs (68/128), including the 4 redone.
- 19:32-20:25 Chunk 3: 36 runs (104/128).
- 20:25-21:04 Chunk 4: 24 runs. **Grid complete: 128 of 128 runs, 0 missing, 12,067 s cumulative** (3.35 h; would also have fit the original 4 h cap).
- 21:04 `analyze.py --arm small --stage 1` (38 s): G2 K law chosen on fit rows only, M_K_coef (pooled fit RMSE 0.574 vs 0.914 for M_K_exp). **forecasts.csv SHA-256 `3dad89e9b0f65e0dd2afcdbf4dfd220eec96516df2f1513a9e87eaf090466ed7`.** Held-out rows were read only after this line.
- 21:05 `analyze.py --arm small --stage 2` (26 s): G1 SUPPORTED, G2 SUPPORTED, G3a NOT_ESTABLISHED, G3b NOT_ESTABLISHED (C3 unresolved), G4 NOT_ESTABLISHED, G5 SUPPORTED. Independent fitter agrees.
- 21:05 `verify.py --arm small --output results/small --workers 4` started.
- 21:10 `report.py` wrote RESULTS_ginc_hmm.md and ginc_hmm_results.png/.pdf. The E panel was changed to a linear axis with the gated ε* point (reporting only).
- 22:26 `verify.py` finished: **PASS** in 4,875 s (81 minutes). Training sources match manifest.json. 20 of 20 concepts regenerate identically, ranks recertified, K_bits recomputed, held-out documents regenerate. Oracle and mixture code lengths agree with the independent forward algorithm to 2.5e-13 bits. 128 training streams regenerate (hash and counts). 640 checkpoints replayed in float64 with a maximum difference of 0 bits. 96 law fits reproduce to 1e-4, the forecasts hash is identical, gate verdicts are identical on rerun, and the independent fitter agrees. No verifier fixes were needed.
- 22:28 Unit tests rerun after the grid: 21 passed (34 s).
- 22:29 `report.py --notes results_notes.md --tldr ...` wrote the final RESULTS_ginc_hmm.md and ginc_hmm_results.png/.pdf.
- 22:30 `tar -cf results/small/checkpoints_ginc_hmm.tar checkpoints`: 914,298,880 bytes, SHA-256 `8de2986bbd983280a4e3a9b879d2dd86b4753a93c3b1604f9f965ce534ad8b66`. ARTIFACT_SHA256.json records the SHA-256 of all 810 files under expt_v1/results/, including the tar and every checkpoint. The checkpoints, the tar and the oracle/mixture npz copies stay out of git.
- 22:31 Results committed locally on `results/cpu-expts-11-15` (not pushed; the parent task pushes).

### Wall time

Code and tests about 40 minutes, check-only about 5 minutes, pilot 29 minutes (1,720 s), gated grid 12,067 s cumulative (3 h 22 min wall, 4 chunks), analysis about 1 minute, verification 4,875 s, test rerun 34 s. Artifacts 0.91 GB, under the 2 GB cap.

### 13-GPU launch (not done here)

The freeze SHA for the GPU pilot is **19a77724** (branch `results/cpu-expts-11-15`; it must be pushed first, and launch/README.md assumes branch `exp13-ginc-hmm`). Before any GPU launch, the GPU code path has to be written: `run.py --arm gpu` currently exits with "pending, not run", and the `--pilot-job`, `--pilot-select`, `--merge` and `MODE=verify` paths of launch/README.md do not exist yet; then run the GPU `--check-only`. After that, follow launch/README.md steps 1-7 with `<sha>` = the freeze SHA for the pilot and the "GPU pilot disclosure" SHA for the shards.

### Deviations

Listed in RESULTS_ginc_hmm.md (Deviations): wall cap raised before the freeze; 4 resume-safe chunks with chunk 1 killed by the command timeout (4 runs redone, run_state entry reconstructed); background CPU load; ε_res set at the freeze; report.py E panel changed after the analysis; 13-GPU not run and its code path not written; not pushed.
