# G1 seeds3 regrade: PARTIAL, 7 of 16 runs (Sherlock array 48812674, tasks 0-6)

Qwen2.5-0.5B, re-graded from saved final weights with `regrade.py --engine pilot`; accuracy on GSM8K and MATH-500.
The other 9 runs (tasks 7-15) are still pending (tasks 7-9 submitted as job 48821682); this set is incomplete and
nothing here is an analysis, only the graded numbers read from the Slurm `.out` files in `slurm_logs/`.

| task | run | GSM8K | MATH-500 |
|---|---|---|---|
| 0 | base (no fine-tuning) | 0.2873 | 0.200 |
| 1 | L1 gradeschool 0.5 + L3 olympiad 0.5 [answer_only], lr 5e-6, s0 | 0.0864 | 0.086 |
| 2 | same mix, s1 | 0.0834 | 0.074 |
| 3 | same mix, s2 | 0.0872 | 0.078 |
| 4 | L1 gradeschool only, lr 2e-5, s0 | 0.4344 | 0.220 |
| 5 | same, s1 | 0.4071 | 0.216 |
| 6 | same, s2 | 0.3889 | 0.226 |

Open question for review: the mixed runs score far below the base model on GSM8K (about 0.085 vs 0.287); check the
regrade answer format (answer_only) before reading anything into it. Per-run grading output is in each
`regrade_pilot/` directory. The regrade report json on Sherlock is the stale 0/16 version and is not used.
