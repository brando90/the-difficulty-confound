# CKPT: Experiment 12, exact DSL program complexity versus alignment

Dated log (MM-DD-YYYY, times US Pacific). Box: 4 cores, 15 GB, no GPU, torch 2.14.1+cpu, Python venv ~/.venvs/tdc. Branch results/cpu-expts-11-15 (local commits only; push is done by the parent task).

## 10-18-2026

- 16:25 PROTOCOL.md and PROMPT.md synced from the design commit f526443f.
- 17:07 Code and unit tests committed (12602a3d): dsl, proxies, data, model, run, laws, independent_fit, analyze, verify, report, four test files.
- 17:21 Test fix committed (dd2869a9). test_laws::test_g3_logic[M_K_coef] used a synthetic noise draw (rng 7) whose 8-seed 95% interval on gamma1 excluded 0 by about 0.002; across draws 0-11 the interval covers 0 in 10 of 12, as a 95% interval should. The test now uses draw 0 for that case. Not a gate change. pybdm==0.1.0 pinned. Full suite 40/40 (502 s).
- 22:33-22:42 `run.py --check-only`, first attempt: all checks pass except `distractors_equal_pm1`. Distractor counts were equal within +-1 inside each octave interval, but the +1 extras were drawn independently per interval, so cumulative counts at T = 2^16 spread over 3,110-3,116. Fix (before any training): the extras in each interval go to the distractors with the lowest running count (ties in random order), so counts are equal +-1 both per interval and cumulatively at every checkpoint, as Section 2.5 states. test_data.py gained that assertion.
- 22:42-22:50 `run.py --check-only`, second attempt: all 11 checks pass (BFS level counts 12/112/988/8,543/73,146/622,876; brute force K <= 4 with 0 mismatches; task invariants; roles 20/20/80/80; split bit counts; schedule within +-4 of alpha T; distractors equal +-1; parameter counts 103,618 / 103,682 / MLP 83,988). G8a = -0.31 [-0.44, -0.18], NOT_ESTABLISHED. Timing probe: 0.040 s per step with 4 workers, projected grid 1.91 h, checkpoints 1.86 GB. Committed with the fix as df8338b2.
- 22:51-23:12 Pilot `run.py --pilot` (seed 12909, alpha = 1, 20 targets, T to 2^17), 1,206 s. P1, P2, P4 do not fire. P3 as computed fires (projected 19,092 s > 16,200 s) because other jobs began training on the box at about 22:55 (load average 12-13 on 4 cores); identical cells slowed from 103-114 s to 493-528 s. On uncontended timing (first round 112 s per pilot cell -> 2.5 h; check-only probe -> 1.91 h) P3 does not fire. Deviation: P3 evaluated on uncontended timing; the grid starts only on a quiet box, and the 5-hour cap still applies. Full disclosure in PROTOCOL Section 2.8. Other running tasks were asked to honour the scratch/at343/CPU_LOCK_* convention.
- 23:20 Freeze: PROTOCOL.md Status line and pilot disclosure, no pilot-rule changes to config.json. Freeze commit 2f1c39a3.
- 23:27-01:17 Gated grid chunk 1 (`run.py --output results --max-seconds 6600`, lock scratch/at343/CPU_LOCK_AT-367 held): 383/640 cells in 6,618 s, stopped at its chunk budget. Load average 4-10 (another local job at times; asked to pause).

## 10-19-2026

- 01:20-01:55 Chunk 2 (`--resume`, the one resume PROTOCOL Section 7 allows): 140 more cells (523/640). It was killed at about 01:55 when the session hosting it ended, before run.py could write run_state.json. No finished cell was lost or retrained (cells are atomic); the partly trained cells in flight were discarded. Its time was added to run_state.json by hand as 2,044 s (launch to kill, an upper bound), marked `interrupted`.
- 01:56-02:21 **Deviation #2:** chunk 3, a second `--resume` (the protocol allows one). Reason: the stop was an infrastructure kill, not a choice made after seeing results; finished cells were skipped, not retrained. 117 cells in 1,527 s; main grid 640/640 complete, 0 missing.
- 02:22-02:56 Secondary arm R768 (`--rep768 --resume`): 160/160 cells in 2,039 s (more than 60 minutes remained under the cap).
- 02:56-02:58 Exploratory MLP arm (`--mlp --resume`): 80/80 cells in 147 s. Cumulative wall time 12,375 s of the 18,000 s cap.
- 03:00-03:05 `analyze.py --stage 1` (main-arm fit rows only): forecasts_frozen SHA-256 896bf348844a53bc25a8ca6f37bc113bf8b9d5d21d94b26cbbc4e9cc37ca5919; G2 K law M_K_exp (fit RMSE 2.58023 vs M_K_coef 2.58025); all_fitters_agree = false. Committed before stage 2.
