# g1 3-seed check (Sherlock, H100, 10-20/21-2026): as-run results; the grader gate FAILED

**TLDR:** All 16 runs finished (array 48737606, exit 0 each, 2.04 H100-hours of task wall time by sacct; the report's own sum of per-run wall seconds is 1.92). The pre-registered grader check (`../../CKPT.md`, "g1 3-seed check") **failed**: the new base-model evaluation (fast batched grader, torch 2.6.0) differs from the pilot's base evaluation by +1.52 points on GSM8K (95% CI [0.0, 3.03]; |delta| > 1.5) and +1.8 on MATH-500 (CI [-0.6, 4.2]; |delta| > 1.5). The pre-registration says that on a fail every run is re-graded from its saved weights with the pilot's engine and **only that is read**. So the tables below are descriptive and are **not** the pre-registered reading. The re-grade is Sherlock array 48805697 (report job 48805698, submitted 10-21-2026 07:09); its results go in `results_3seed/regrade/` and `../../CKPT.md` when it ends.

Gate numbers (`gate.json`, from `gate_check.py`; `seeds3_report.py` skipped the check because `results_pilot/runs/runs/*` has no `DONE` markers, so `gate_check.py` runs the same code against a copy with the markers added):

| set | pilot base | new base | new - pilot | 95% CI over items | item agreement | pass |
|---|---|---|---|---|---|---|
| GSM8K | 29.34 | 30.86 | 1.52 | [0.0, 3.03] | 92.27% | no (delta 1.52 > 1.5) |
| MATH-500 | 18.4 | 20.2 | 1.8 | [-0.6, 4.2] | 92.2% | no (delta 1.8 > 1.5) |

The GSM8K interval just touches 0 and the delta misses the cap by 0.02 points; MATH-500's interval includes 0 but its delta is 1.8. Seed 0 of E and H against the pilot's seed-0 runs (same run ids; GPU training is not bit-reproducible): GSM8K 41.85 -> 42.38 (E, 5e-6), 44.20 -> 43.52 (E, 2e-5), 44.43 -> 44.50 (H, 5e-6), 39.50 -> 38.82 (H, 2e-5); every one within 0.7 points.

## As-run per-run table (fast grader; descriptive only)

| run | GSM8K | MATH-500 | L4-5 | stopped % | boxed % | mean tokens | L1 bpb | L3 bpb | HellaSwag |
|---|---|---|---|---|---|---|---|---|---|
| L1_gradeschool1__T152x32x2048__lr5e-06__s0 | 42.38 | 27.6 | 12.6 | 98.6 | 96.0 | 223.8 | 0.1344 | 0.3793 | 49.55 |
| L1_gradeschool1__T152x32x2048__lr5e-06__s1 | 41.77 | 27.0 | 13.36 | 98.3 | 96.6 | 222.3 | 0.1344 | 0.3793 | 49.2 |
| L1_gradeschool1__T152x32x2048__lr5e-06__s2 | 44.5 | 26.8 | 14.12 | 98.7 | 96.8 | 220.3 | 0.1344 | 0.3795 | 49.45 |
| L1_gradeschool1__T152x32x2048__lr2e-05__s0 | 43.52 | 21.8 | 9.16 | 98.6 | 97.9 | 215.5 | 0.1365 | 0.3981 | 48.65 |
| L1_gradeschool1__T152x32x2048__lr2e-05__s1 | 40.64 | 24.8 | 9.92 | 98.7 | 97.5 | 216.8 | 0.137 | 0.3994 | 49.25 |
| L1_gradeschool1__T152x32x2048__lr2e-05__s2 | 39.95 | 21.2 | 6.87 | 97.8 | 96.5 | 222.9 | 0.1372 | 0.3995 | 50.05 |
| L3_olympiad1__T152x32x2048__lr5e-06__s0 | 44.5 | 27.2 | 11.07 | 96.1 | 96.1 | 331.5 | 0.2126 | 0.3468 | 49.0 |
| L3_olympiad1__T152x32x2048__lr5e-06__s1 | 43.75 | 26.6 | 13.74 | 96.0 | 95.5 | 342.6 | 0.2125 | 0.3469 | 48.95 |
| L3_olympiad1__T152x32x2048__lr5e-06__s2 | 44.05 | 26.6 | 12.21 | 96.2 | 95.7 | 334.9 | 0.212 | 0.3469 | 49.0 |
| L3_olympiad1__T152x32x2048__lr2e-05__s0 | 38.82 | 20.6 | 9.16 | 94.5 | 94.5 | 341.0 | 0.2275 | 0.3507 | 47.8 |
| L3_olympiad1__T152x32x2048__lr2e-05__s1 | 39.5 | 25.8 | 12.6 | 95.6 | 95.6 | 351.3 | 0.2331 | 0.351 | 49.7 |
| L3_olympiad1__T152x32x2048__lr2e-05__s2 | 40.26 | 21.8 | 11.07 | 94.6 | 94.6 | 342.4 | 0.2316 | 0.3512 | 48.25 |
| L1_gradeschool0.5+L3_olympiad0.5[answer_only]__T152x32x2048__lr5e-06__s0 | 8.72 | 8.0 | 3.82 | 100.0 | 100.0 | 11.0 | 0.238 | 0.3777 | 48.85 |
| L1_gradeschool0.5+L3_olympiad0.5[answer_only]__T152x32x2048__lr5e-06__s1 | 8.87 | 7.4 | 3.44 | 100.0 | 100.0 | 11.0 | 0.237 | 0.3773 | 49.3 |
| L1_gradeschool0.5+L3_olympiad0.5[answer_only]__T152x32x2048__lr5e-06__s2 | 8.42 | 7.6 | 3.82 | 100.0 | 100.0 | 11.0 | 0.24 | 0.3789 | 49.3 |
| base | 30.86 | 20.2 | 8.78 | 93.4 | 2.7 | 211.6 | 0.1948 | 0.3659 | 48.7 |

## As-run readout (fast grader; descriptive only; points; per seed, mean, 95% CI over items; t over seeds)
 (points; per seed, mean, 95% CI over items; t over seeds)

- gsm8k_E_minus_H@5e-06: [-2.12, -1.97, 0.45] mean -1.21 CI [-3.34, 0.78] t [-4.81, 2.38]
- math500L45_H_minus_E@5e-06: [-1.53, 0.38, -1.91] mean -1.02 CI [-4.07, 2.16] t [-4.07, 2.03]
- gsm8k_E_minus_H@2e-05: [4.7, 1.14, -0.3] mean 1.84 CI [0.03, 3.66] t [-4.55, 8.24]
- math500L45_H_minus_E@2e-05: [0.0, 2.67, 4.2] mean 2.29 CI [-0.51, 5.22] t [-2.99, 7.57]
- gsm8k_E@5e-06_minus_F: [33.66, 32.9, 36.09] mean 34.22 CI [31.61, 36.75] t [30.09, 38.35]
- math500_L45_E@5e-06_minus_F: [8.78, 9.92, 10.31] mean 9.67 CI [5.6, 13.74] t [7.7, 11.64]
- gsm8k_E@5e-06_minus_base: [11.52, 10.92, 13.65] mean 12.03 CI [9.48, 14.51] t [8.47, 15.59]
- gsm8k_E@2e-05_minus_F: [34.8, 31.77, 31.54] mean 32.7 CI [30.17, 35.13] t [28.18, 37.22]
- math500_L45_E@2e-05_minus_F: [5.34, 6.49, 3.05] mean 4.96 CI [1.78, 8.02] t [0.62, 9.31]
- gsm8k_E@2e-05_minus_base: [12.66, 9.78, 9.1] mean 10.51 CI [8.09, 12.91] t [5.81, 15.21]
- gsm8k_H@5e-06_minus_F: [35.78, 34.87, 35.63] mean 35.43 CI [32.78, 38.03] t [34.22, 36.64]
- math500_L45_H@5e-06_minus_F: [7.25, 10.31, 8.4] mean 8.65 CI [4.58, 12.72] t [4.82, 12.48]
- gsm8k_H@5e-06_minus_base: [13.65, 12.89, 13.19] mean 13.24 CI [10.69, 15.85] t [12.29, 14.19]
- gsm8k_H@2e-05_minus_F: [30.1, 30.63, 31.84] mean 30.86 CI [28.35, 33.33] t [28.64, 33.08]
- math500_L45_H@2e-05_minus_F: [5.34, 9.16, 7.25] mean 7.25 CI [3.81, 10.81] t [2.51, 11.99]
- gsm8k_H@2e-05_minus_base: [7.96, 8.64, 9.4] mean 8.67 CI [6.24, 11.14] t [6.88, 10.46]
- gsm8k_F_minus_base: [-22.14, -21.99, -22.44] mean -22.19 CI [-24.94, -19.41] t [-22.76, -21.61]
- math500L45_F_minus_base: [-4.96, -5.34, -4.96] mean -5.09 CI [-8.78, -1.53] t [-5.64, -4.54]
- gsm8k_interaction_lr:  mean 3.06 CI [0.78, 5.36]

As-run verdicts from `seeds3_report.py`: sign flip "LR-DEPENDENT, SIGN NOT ESTABLISHED"; P1 (E-H >= +3 at 5e-6) NOT SUPPORTED; H beats format-only on GSM8K at 5e-6: yes. The format-share ratio (-1.68) is not interpretable: the format-only arm F stopped working problems out (mean 11 generated tokens, GSM8K 8.4-8.9%, below base), so it learned a bare boxed answer and F - base is -22 points. F is a "no-reasoning" control here, not a clean format-only control.

## Cost and files

- 16 array tasks (6.5-8.8 min each, base 3.1 min), one H100 (sh04-01n07), `sacct_g1s3.txt`; the array started 10-20 23:52 and ended 10-21 06:47 PDT (queue time, not run time, took most of it).
- `report_as_run.json/.md` (written by the Sherlock report job; recomputed locally from `runs/` with `seeds3_report.py --out-root runs`, identical), `gate.json`, `gate_check.py`, `runs/runs/*/metrics.jsonl` (per-item correctness), `runs/runs/*/generations/`, `slurm_logs/`, `jobs_g1s3.txt`, `inventory_g1s3.txt`, `submit_g1s3.log`. Weights stay on Sherlock (`$SCRATCH/tdc_g1s3/runs/runs/*/final`).
- Re-grade code: `../seeds3/regrade.sbatch`, `regrade_report.sh`, `submit_regrade.sh` (new, added after the gate failed; the pre-registered code files are unchanged).
