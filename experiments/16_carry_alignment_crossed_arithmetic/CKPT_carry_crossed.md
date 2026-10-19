# Checkpoint log: `16_carry_alignment_crossed_arithmetic`

Dates are MM-DD-YYYY. Branch `exp16-carry-crossed`, from main `3f0b6670`.

- 10-19-2026: Environment: Python 3.12.14, torch 2.14.1 (CPU), numpy 2.5.4, scipy 1.18.1 (`expt_v1/requirements.txt`). GPU environment pinned to torch 2.6.0+cu124 (`requirements-gpu.txt` holds the rest).
- 10-19-2026: Built `expt_v1/`: carry.py (exact dynamic program and sampler; natural shares match PROTOCOL 2.3 to 4 decimals; 1–4 digits match brute force), mlp_arm.py, transformer_arm.py, run.py, laws.py, independent_fit.py, analyze.py, verify.py, report.py and tests.
- 10-19-2026: Unit tests: 14 pass (`python -m pytest -q expt_v1`), including synthetic end-to-end analysis dry runs for both arms and a CPU smoke run of the 16b loop under torch 2.14.1 and 2.6.0.
- 10-19-2026: First synthetic dry run: the independent fitter disagreed with laws.py on one `M_align_eff` fit (held-out RMSE 0.5545 vs 0.5536; costs 20.9423 vs 20.9429) because the N term leaves a nearly flat direction. Fixed before the freeze by crossing the start grid with beta in {0.1, 0.5, 2.0}; all dry runs then AGREE.
- 10-19-2026: `run.py --arm mlp --check-only` and `--arm transformer --check-only` pass (108 training mixes exact, 2,880 / 18,000 test pairs, DP total 10**16, parameter counts as configured, 240 distinct shards).
- 10-19-2026: 16a pilot (seed 16909, 6 runs, 8 s). The epsilon* rule fired: cascade at alpha = 0.1 is 4.92 bits at 1,000 updates, so 16a epsilon* = 4.95 bits. Disclosed in PROTOCOL Section 2.1.
- 10-19-2026: PROTOCOL pre-freeze changes drafted for sign-off: pilot disclosure, stream blocks, censoring and starts, alignment with Experiment 11's G1–G4, hardware decisions. Not frozen yet.
