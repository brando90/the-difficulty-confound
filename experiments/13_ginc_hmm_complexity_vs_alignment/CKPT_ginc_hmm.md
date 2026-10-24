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
- 10-19-2026 review fix: report.py Section 6 rows now state what this run found (η check for the G1/G4 row; G5 as a position split only, not evidence for identification), the takeaway glosses its symbols, and the TLDR matches. Reporting only; no number changed.

## 13-GPU (Laolu Oguneye; branch `exp13-ginc-gpu`)

### 10-20-2026

- 11:50–12:30 13-GPU code path written: `gpu_arm.py`, `analyze_gpu.py`, `verify_gpu.py`, `test_gpu.py`, the full N laws in `laws.py` and `independent_fit.py`, `--arm gpu` dispatch in `run.py`, `analyze.py` and `verify.py`, the GPU keys in `config.json` (optimizer, distractor evaluation documents, pilot α and tie rule, β starts, extension cap; `gpu_torch_version` 2.6.0), `launch/sherlock_prep.sbatch`, and launcher fixes (torch 2.6.0 cu124/cu126, system git module on Sherlock, GPU constraint, one-time prep). 30 of 30 unit tests pass.
- 11:57 First `run.py --arm gpu --check-only` failed: `levels cannot supply 5 concepts within the candidate cap: {128: 0}` (64: 18 calibration candidates, 128: 0; band widened to ±0.30). A diagnostic on an independent stream gave about 0.07% strongly connected and aperiodic at n = 128, with about 2 in 10 of those at full rank. The fix is the per-level extension stream (PROTOCOL Section 7, 13-GPU implementation notes). The 13-small concepts regenerate identically with the changed `build_concepts`.
- 12:10–12:15 `run.py --arm gpu --check-only --workers 3` on the laptop (numpy 2.5.4): PASS in 320 s. 25 concepts, all with certified rank = n, $h_{\text{ref}}$ 2.862, band ±0.30. n = 128 extension: 13,799 drawn, 5 passed checks 1–2, 5 accepted (indices 5800, 8256, 8967, 13314, 14798). Entropy rates 2.68–3.07 bits. Parameter counts S/M/L match. Checkpoints: S, M 2^22…2^30; L 2^22…2^28 and 2^29 (FLOP cap). 60 α schedules exact over 8,388,608 documents. 0 of 31,360 evaluation documents among 6,553,600 training documents. 360 shards. ε_id over all positions = log2(1/α)/128 for every target (0.0338 bits at α = 0.05). Receipt: `expt_v1/results_check_gpu.json`.
- 12:20 Same check with numpy 2.2.6, scipy 1.16.3, torch 2.6.0 (the GPU environment): concepts, evaluation documents, oracle.npz and mixture.npz bit-identical; only the logged entropy rates in tasks.json differ (by at most 4.4e-16).
- 12:25 Smoke tests, not results: a tiny CPU pilot job (S, T to 2^18) ran end to end. A CPU loss trace of S on the pilot concept: about 243 bits/token at step 1 (model.py's default N(0,1) tied-embedding init), 12 bits at 2^20 tokens, 4.45 bits training loss at 2^22 (the first checkpoint). L may still be in this transient at 2^22; the pilot shows it. Synthetic 13-GPU results through `analyze.py --arm gpu`: the pooled N law recovered the planted κ 0.70, γ1 −0.051, η 0.80, β 0.50; the independent fitter agrees.

### 10-23-2026

- Decisions signed off by Laolu (PROTOCOL Section 7, 13-GPU implementation notes): the n = 128 extension, the gate definitions and every-seed rule, the N-law details, the commit order, the environment, and the branch name. Branch updated to main 6f66bd3f. The "Experiment 13 GPU code" commit follows; the pilot runs at it.
- 15:08–15:25 SNAP. skampere2 is reserved for other work, so 13-GPU runs on **skampere1** (8 A100-SXM4-80GB, all shared with other users). GPUs are used only where at least 24 GB is free. `/dfs/scratch0`, `/dfs/scratch1` and the node-local disk were 99–100% full, so the clone is sparse (this experiment's folder only, 221 files; the full checkout on DFS took more than 25 minutes and was abandoned), the virtual environment lives on node-local disk (`/lfs/skampere1/0/laoluog1/venvs/tdc_exp13`: torch 2.6.0+cu126, numpy 2.2.6, scipy 1.16.3; 30 of 30 unit tests pass), and shard outputs go to `/dfs/scratch1/laoluog1/tdc_exp13/runs/gpu_v1`.
- 15:53 `launch/snap.sh` exited silently: `nvidia-smi | head -n 1` exits 141 (SIGPIPE) under `set -o pipefail`. It had never run before. The pilot was started with `launch/snap_autopilot/pilot_launch.sh` (same checks, claims, logs and status format) at 2c625111, unmodified. snap.sh is fixed in this commit (`|| true` on the two pipes); the run did not use it.
- 15:54–16:01 `run.py --arm gpu --check-only` on skampere1: PASS. Receipt kept as `runs/gpu_v1/logs/results_check_gpu_skampere1.json`.
- 16:01–16:12 **GPU pilot**, 9 of 9 runs finished (exit 0), 3 at a time on GPUs 1, 3 and 6. Excess ε at 2^26 (bits/token): S 0.5× 0.0624, 1× 0.0524, 2× 0.0491; M 0.5× 0.4207, 1× 0.0577, 2× 0.0534; L 0.5× 0.0963, 1× 0.0759, 2× 0.0559. Frozen rule: **S, M and L all 2×** (lr 2e-3, 1.2e-3, 6e-4). The 2× multiplier is the top of the frozen grid for every size, so a larger rate might do better still; the rule allows nothing else, and this is reported as a limitation. L starts near 681 bits/token (default init), and by 2^22 its excess is 0.8–2.0 bits, so the initialization transient is over by the first checkpoint.
- 16:12 **"Experiment 13 GPU pilot disclosure" commit 2c3d7e8f** (config gpu.lr_selected {S: 2, M: 2, L: 2} plus the PROTOCOL Section 7 paragraph only), made by `launch/snap_autopilot/autopilot.sh` in the skampere1 clone, fetched and pushed from the laptop with the same SHA. Every shard runs at 2c3d7e8f.
- 16:12 The autopilot halted before launching shards: `--check-only` had rewritten the tracked receipt `expt_v1/results_check_gpu.json`, so its clean-tree check failed. Lost: 2 h 2 min. Fixed at 18:14 by restoring the tracked file (the skampere1 receipt is kept in the logs folder, above) and restarting the autopilot. Nothing else changed.
- 18:14 **Shards started**: 4 workers on GPUs 3, 5, 6 and 7 (at least 24 GB free at launch), L shards first (240–359, then M, then S). Each worker waits while its GPU has under 24 GB free and verifies each shard on its GPU right after training (`VERIFIED`). L trains at about 200,000 tokens/s.
