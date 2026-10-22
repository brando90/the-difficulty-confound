# g1 3-seed check, re-graded with the pilot engine (the pre-registered readout)

Qwen2.5-0.5B, E = easy-only (L1 grade-school), H = hard-only (L3 olympiad), F = format-only (answer_only mix), 3 seeds, lr 5e-6 and 2e-5.
The as-run grader failed the gate (see `README.md`), so per CKPT.md only this re-grade (saved weights, pilot decoding engine) is read.
16/16 re-grades finished (sacct: all COMPLETED, exit 0). Full report: `g1s3_regrade_report.md/.json`.

## Takeaway
Hard-to-easy transfer is real but not stronger than easy-to-easy: H lifts GSM8K by +15.1 points over the base model (CI [12.6, 17.6]) and beats F by +35.2,
but E minus H on GSM8K is -0.94 [-2.98, 1.09] at 5e-6 (no difference; the pilot's >=3 point E-H gap is NOT supported) and +1.87 [0.03, 3.69] at 2e-5.
The pilot's sign flip with lr is LR-dependent but not established over seeds (interaction +2.81, item-bootstrap CI [0.66, 5.03]; the seed-level t intervals are wide).

## Per-arm table (percent; mean ± 95% t-interval over 3 seeds; base model GSM8K 28.73, MATH-500 20.0, L4-5 8.4)
| arm | lr | GSM8K per seed | GSM8K mean ± 95% CI (t, 3 seeds) | MATH-500 mean ± CI | MATH-500 L4-5 mean ± CI |
|---|---|---|---|---|---|
| E (easy-only) | 5e-06 | 42.08, 42.53, 43.97 | 42.86 ± 2.45 | 26.73 ± 1.03 | 12.60 ± 0.96 |
| F (format-only) | 5e-06 | 8.64, 8.34, 8.72 | 8.57 ± 0.50 | 7.93 ± 1.52 | 3.94 ± 2.39 |
| H (hard-only) | 5e-06 | 44.12, 43.44, 43.82 | 43.79 ± 0.85 | 26.73 ± 1.25 | 14.00 ± 3.59 |
| E (easy-only) | 2e-05 | 43.44, 40.71, 38.89 | 41.01 ± 5.69 | 22.07 ± 1.25 | 8.78 ± 4.35 |
| H (hard-only) | 2e-05 | 38.59, 38.51, 40.33 | 39.14 ± 2.56 | 21.87 ± 5.45 | 9.80 ± 1.98 |

## Contrasts (points, mean over seeds, 95% CI over items)
- E-H GSM8K: 5e-6 -0.94 [-2.98, 1.09]; 2e-5 +1.87 [0.03, 3.69]; interaction with lr +2.81 [0.66, 5.03]
- H-E MATH-500 L4-5: 5e-6 +1.40 [-1.78, 4.58]; 2e-5 +1.02 [-1.91, 4.07]
- E - F GSM8K: 5e-6 +34.29 [31.72, 36.85]; 2e-5 +32.45 [29.87, 34.93]. H - F GSM8K: 5e-6 +35.23 [32.55, 37.86]; 2e-5 +30.58 [28.13, 32.95]
- E - F MATH-500 L4-5: 5e-6 +8.65 [4.83, 12.34]; 2e-5 +4.83 [1.65, 7.89]. H - F: +10.05 [5.98, 14.12]; +5.85 [2.29, 9.41]
- E - base GSM8K: +14.13 / +12.28; H - base: +15.06 / +10.41
- Verdicts: sign_flip LR-DEPENDENT, SIGN NOT ESTABLISHED; P1 (E-H >= 3 at 5e-6) NOT SUPPORTED; H beats F on GSM8K at 5e-6: YES.

## Caveats
- Format-only arm F is not a clean format control: it collapsed to bare boxed answers (GSM8K 8.6%, 20 points below base), so "H beats F" is nearly trivial and `format_share_of_H_gain` (-1.34) is not meaningful.
- Gate on the re-grade (`gate_regrade.json`, `gate_check_regrade.py`): base re-graded 28.73 vs pilot 29.34 GSM8K (-0.61, CI [-2.05, 0.83], pass); MATH-500 20.0 vs 18.4 (+1.6, CI [-0.8, 4.2]), 0.1 over the 1.5 cap, so strictly it still fails on MATH-500.
  Seed-0 re-grades vs pilot runs agree within 0.9 (GSM8K) and 1.2 (MATH-500) points; item agreement for the base is 92%, so some residual engine/torch-version noise remains (torch 2.9.1 -> 2.6.0).
- 3 seeds: seed-level intervals are wide; most E-H contrasts include 0.

## Compute (sacct, Sherlock H100)
Re-grade jobs 48812674 (0-6), 48821682 (7-9), 48901686 (10-15), report 48901689: 1.63 H100-hours. As-run training+grading array 48737606: 2.04 H100-hours. Total 3.67.
