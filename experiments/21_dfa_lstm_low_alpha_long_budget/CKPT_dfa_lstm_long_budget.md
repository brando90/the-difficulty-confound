# CKPT: Experiment 21, floor or late learning at α ∈ {0.05, 0.15} with the LSTM

Dated log of every step of the Experiment 21 run on the CPU box (4 logical cores, 15 GB RAM, no GPU), branch `results/cpu-expts-11-15`. Times are US Pacific. Commands run from `experiments/21_dfa_lstm_low_alpha_long_budget/expt_v1/` with the pinned environment of requirements.txt (Python 3.12.3, torch 2.14.1+cpu), the same as Experiments 11 and 20.

## 10-20-2026

- 01:25 Working clone at 50da25cf (Experiment 20 results). Experiment 19 is the GPU transformer rerun, so this is Experiment 21.
- 01:30 Code copied from Experiment 20's expt_v1. `dfa.py`, `data.py`, `model.py` and `requirements.txt` are byte-identical (Experiment 20's 37 unit tests cover them; receipt in its `pytest_receipt.txt`). `run.py` changes only its docstring, the source list, the cell order (α = 0.15 first, n ascending) and the removal of the pilot, calibration and LSTM-arm paths. `config.json`: α ∈ {0.05, 0.15}, checkpoints 2^9 … 2^18, wall cap 43,200 s, analysis seed 21102026, the verdict thresholds, and the law and gate keys of Experiment 20 removed. `analyze.py` and `verify.py` are new.
- 01:31 `run.py --check-only` passes all checks (Hopcroft = Moore for all 32 tasks, test set, no test strings in pools, schedule within 1 example of αT at every checkpoint, no pool reuse, 84,290 parameters). Timing probe under a shared box (Experiment 14's resume was running): 0.071 s per step, 1.09 s per evaluation, about 300 s per cell, grid 5.4 h. Receipt in results_check.json. tasks.json and all 64 data/*.npz files regenerate byte-identical (SHA-256) to Experiment 20's.
- 01:33 Per-length label entropy of the 16 targets computed from the test labels (no training): 0.929 to 0.997 bits.
- 01:35 `assemble_mix` checked on 7 cells: the task-slot sequence through 2^16 equals Experiment 20's, the lengths and pool order do not (PROTOCOL Section 3.2, item 3). `verify.py` checks all 256.
- 01:44 Pipeline smoke test in a scratch copy (1 seed, budget 2^13, not part of the results): run, `analyze.py --distractors`, `analyze.py` and `verify.py` ran end to end; `verify.py` PASS after fixing its slot-prefix comparison to slice both sides. Nothing from the smoke test is used.
- 01:46 **Freeze commit, "Freeze Experiment 21 protocol".** From here on Sections 2–6 of PROTOCOL.md, `config.json`, `run.py`, `analyze.py` and `verify.py` are read-only. Its hash is the commit that adds this line (see `git log`). The grid waits for the CPU box: Experiment 14's resume (AT-527) holds it.
- 04:17 Grid launched: `run.py --output results --max-seconds 7200` under scratch lock CPU_LOCK_AT-540. Experiment 17b (AT-453) started on the same 4 cores 6 s earlier and ran until about 06:20 (deviation 1).
- 05:15 Chunk 1 stopped at its time budget: 28 of 256 cells in 3,473 s (about 450 s per cell with the box shared). Chunks from here on use `--resume --max-seconds 3300`.
- 06:13 Chunk 2: 36 cells in 3,479 s; 64 of 256.
- 06:35 Chunk 3 killed mid-run (24 cells done in 1,211 s) when the worker running it stopped; not in run_state.json (deviation 2). Cells in flight were redone by `--resume`.
- 07:32 Chunk 4: 72 cells in 3,387 s; 160 of 256.
- 08:15 and 08:20 Brando: compute should not go through Rylan, reruns go to Sherlock CPU nodes, and no more LSTM work beyond finishing this run. The last 96 cells finished on this box as the cheapest way to finish.
- 08:32 Chunk 5: 64 cells; 224 of 256.
- 09:33 Chunk 6: 20 cells; 244 of 256. The n = 32, α = 0.05 cells took about 815 s each (deviation 3).
- 10:48 Chunk 7: 12 cells. **Grid complete: 256 of 256 cells, 0 missing, about 21,800 s cumulative** (cap 43,200 s).
- 11:54 `analyze.py --distractors` wrote results/distractors.json (256 cells, 2^16 and 2^18).
- 11:55 `analyze.py` wrote results/analysis.json. Overall verdict **LATE_LEARNING**; cells: (4, 0.15) and (8, 0.15) LATE_LEARNING, (16, 0.15), (4, 0.05), (8, 0.05), (16, 0.05) UNRESOLVED, (32, 0.15) and (32, 0.05) FLOOR_HOLDS.
- 12:28 `verify.py`: **PASS** in 1,966 s (all 8 checks; 2,560 checkpoints replayed with max |Δ| = 0; retrained cell identical; 256 of 256 distractor cells replay identically).
- 12:35 `report.py --notes results_notes.md` wrote RESULTS_dfa_lstm_long_budget.md and the four-panel figure (PNG, PDF). report.py was written at 06:15 during the grid and only formats (deviation 4).
- 12:40 `tar -cf results/checkpoints_dfa_lstm_long_budget.tar -C results checkpoints`: 870,451,200 bytes, SHA-256 `f620f02a1d790116ac1b77f966bf50a5f8eed6d527b9e96cc7e65c2b8af5b9c8`. ARTIFACT_SHA256.json records the SHA-256 of all 3,146 files under expt_v1/results/, including the tar and every checkpoint. The tar and checkpoints/ stay out of git.
- 12:45 Committed on `results/cpu-expts-11-15`.
