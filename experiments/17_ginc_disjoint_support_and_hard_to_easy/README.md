# Experiment 17: Putnam vs GSM8K, synthetic (hard-to-easy transfer), plus disjoint support

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/17_ginc_disjoint_support_and_hard_to_easy/README.md>

Experiment 17 asks two questions Brando raised on 10-18-2026 about the complexity-versus-alignment program, on Experiment 13-small's GINC-style HMMs and model (CPU).

- **Arm H (primary), Putnam vs GSM8K:** does training only on hard concepts (32-state HMMs) buy easy ones (4-state HMMs from the same family) as well as training on easy concepts does, at the same compute? If not, difficulty is not a total order.
- **Arm D (secondary), Putnam vs poetry:** the same training pools on a disjoint token alphabet. Does such data buy anything, and does alignment finally dominate complexity (elasticity ratio E > 1) under genuine mismatch? Experiment 13 measured E = 0.80 within one alphabet.

Design and gates: [PROTOCOL.md](PROTOCOL.md). Results: [RESULTS_exp17.md](RESULTS_exp17.md). Run log: [CKPT_exp17.md](CKPT_exp17.md). Code: [expt_v1/](expt_v1/).

**Experiment 17b (follow-up):** arm H rerun with coverage-matched easy pools. Design: [PROTOCOL_exp17b.md](PROTOCOL_exp17b.md). Results: [RESULTS_exp17b.md](RESULTS_exp17b.md). Code: [expt_v2_coverage/](expt_v2_coverage/) (`python run_v2.py --output results && python analyze_v2.py && python report_v2.py && python verify_v2.py`). Answer: once coverage is matched, hard-only no longer beats easy-only zero-shot on easy targets (gap closed ≈ 100%).

Reproduce (from `expt_v1/`, with torch 2.x CPU, numpy, scipy, matplotlib, pytest):

```
python -m pytest -q                      # unit and synthetic-analysis tests
python run.py --check-only               # concepts, references, invariant checks
python run.py --pilot                    # disclosed pilot
python run.py --output results           # 96 gated runs, resume-safe
python analyze.py && python report.py && python verify.py
```
