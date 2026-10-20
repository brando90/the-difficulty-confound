# CKPT: Experiment 20, exact DFA complexity versus data alignment with an LSTM learner

Dated log of every step of the Experiment 20 run on the CPU box (4 logical cores, 15 GB RAM, no GPU), branch `results/cpu-expts-11-15`. Times are US Pacific. Commands run from `experiments/20_dfa_lstm_complexity_vs_alignment/expt_v1/` with the pinned environment of requirements.txt (Python 3.12.3, torch 2.14.1+cpu), the same as Experiment 11.

## 10-19-2026

- 22:00 Working clone at 3d779865 (Experiment 15 results). Brando approved this rerun on 10-18-2026 ("Go").
- 22:03 Code copied from Experiment 11's expt_v1. dfa.py, data.py, laws.py, independent_fit.py, the three DFA, data and law test files, and requirements.txt are byte-identical. Changed lines against Experiment 11: analyze.py 2 (docstring), run.py 22, model.py 10, verify.py 19, report.py 8, test_model.py 18 and config.json 6. config.json adds `"arch": "lstm"` and the path of Experiment 11's arm metrics, and raises `max_wall_seconds` from 18,000 to 28,800. Every gate value, threshold, seed, α, checkpoint and ε* level is unchanged.
- 22:11 Unit tests: 37 passed in 620 s (pytest_receipt.txt), including LSTM determinism and the npz round trip.
- 22:12 `run.py --check-only` passes all checks: Hopcroft equals Moore for all 32 tasks; tasks pairwise distinct and not complements; a 4,096-string test set; no test strings in pools; schedule within 4 examples of αT; no pool reuse; parameter counts 84,290 (LSTM) and 104,002 (transformer). Timing probe: 0.050 s per LSTM step and 0.95 s per evaluation under 4 workers, about 59 s per cell, so the grid projects to 2.1 h. Receipt in results_check.json. tasks.json and all 64 data/*.npz files are byte-identical (SHA-256) to Experiment 11's results.
- 22:12 ε* rule (PROTOCOL Section 3.3): `eps_rule_from_exp11_arm.py` gives medians over targets of the 8-seed geometric-mean T* of 24,864 (0.1 bits), 16,947 (0.3) and 13,624 (0.5), with 0 of 16 targets at or below 0.1 bits at 2^10. P1 and P1b fire and P2 does not, so the primary ε* is 0.5 bits, exactly as in Experiment 11's frozen config. Receipt in eps_rule.json.
- 22:13 **Freeze commit, "Freeze Experiment 20 protocol".** From here on, Sections 2-5 of PROTOCOL.md and every gate value in config.json are read-only. Its hash is the commit that adds this line (see `git log`).
