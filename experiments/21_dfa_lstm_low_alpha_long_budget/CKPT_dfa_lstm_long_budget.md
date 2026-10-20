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
