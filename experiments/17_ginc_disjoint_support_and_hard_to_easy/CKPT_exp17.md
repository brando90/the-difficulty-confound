# Run log: Experiment 17

- 10-18-2026 22:49 PT: task AT-437 started. Design from Brando's WhatsApp messages (22:47 Putnam/poetry; 22:49 Putnam vs GSM8K; 22:53 "let's do a equivalent putnam vs gms8k but in the synthetic setting", which makes arm H primary). Folder 17 because 16 already holds the carry-chain design.
- 22:56–23:03: code written (copied from Experiment 13-small: hmm.py + regions, model.py, laws.py; new run.py, analyze.py, report.py, verify.py, tests). `run.py --check-only` passed: 50 concepts, entropy band 0.15 bits around h_ref 2.809, all process ranks = n, far = near shifted (identical entropy, disjoint tokens), α schedules exact, pools closed, 0 eval/train document collisions, 96 expected runs.
- 23:03: pilot started (seed 17909, 9 runs, 3 workers). The box carried a background load of about 9–16 from other jobs, so throughput is lower than Experiment 13's.
- 23:27: pilot done (9 runs, 1,429 s, no divergence). Median 5,611 train tokens/s per worker; 21 s per target-checkpoint evaluation; projected grid 3.83 h (primary, 7 checkpoints) under load ~10, so no fallback. ε* rule fired: n = 32 pilot learning term at near α = 0.4, T = 2^19 = 1.128 bits → ε* = 1.13 (Experiment 13 had 1.10). Pilot-informed change before the freeze: 2^14 and 2^15 sat in the initialization transient (excess 40–57 bits/token at 2^14), so the gated checkpoints are 2^16 … 2^20 as in Experiment 13 (fallback 2^16 … 2^19).
- Freeze: PROTOCOL.md, config.json and code committed before any gated run (commit below).
- 10-19-2026 03:20: grid launched after Experiment 12's training finished (02:21); held scratch/at343/CPU_LOCK_AT-437. Chunk 1: 48/96 runs, 6,211 s. Chunk 2 (resume-safe, as designed): 48/96, 5,199 s. Cumulative 11,410 s of the 21,600 s cap. Background load 5–9, no divergence.
- 06:40: analyze.py (6 s): H1 NOT_ESTABLISHED, H2 NOT_ESTABLISHED (easy targets α=0: hard pool better, −3.96 bits/token; α=0.05 n.s.; hard targets α=0: hard better +1.85; α=0.05: easy better −0.24). D1 SUPPORTED (+25.06), E_near 0.801, E_far 0.734, D2 NOT_ESTABLISHED (reverse), D3 NOT_ESTABLISHED (−0.067). report.py, RESULTS_exp17.md, verify.py: VERIFY PASS (12/12 checks, replays exact to 0 bits).
- Post-hoc (disclosed in RESULTS): n = 4 concepts emit 11–15 of 32 tokens, n = 32 emit 31–32, so difficulty and token coverage are confounded in arm H's zero-shot contrast.

## Experiment 17b (AT-453): coverage-matched easy pools

- 10-19-2026 06:47: task AT-453 started (monitor follow-up of AT-437). Named 17b in this folder because 18 (vision) and 19 (GPU) exist.
- 07:23: expt_v2_coverage/run_v2.py --check-only: cov32 (8 n=4 HMMs, 8 tokens/state partitioning all 32) needed entropy band 0.50 (rates 3.02-3.29); cov16 (Exp 17 family, 16 tokens, complementary pairs) band 0.15. All checks pass, v1 concepts unchanged, eval/train disjoint, smoke run finite. First attempt used the simulated quick entropy and scanned 1,500 candidates per slot at band 0.15 for 30+ min; replaced by the exact rate (states emit disjoint tokens), stopped and rerun (1.6 min).
- analyze_v2/report_v2 tested on a stand-in metrics file built from v1 rows (renamed pools); reproduces Exp 17's numbers.
- Freeze: PROTOCOL_exp17b.md, config_v2.json and code committed before any gated run. Grid waits for scratch/at343/CPU_LOCK_AT-439 (vision) to clear.
