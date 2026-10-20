# Experiment 14: does Task2Vec track exact complexity?

**TL;DR.** Not on these probes: the Task2Vec norm does not track exact K (Gates 1-3 NOT_ESTABLISHED in both families); it tracks how well the head already fits a task, and falls as T* rises. Run past the 4 h cap after Gates 1-3 were seen (Deviation 3): the diversity coefficient does not track mixture entropy (Gate 4) and label noise lowers the norm in both families (Gate 5), both NOT_ESTABLISHED.

## Takeaway

On these small probes, the Task2Vec norm does not track exact complexity K. The DFA result even goes the wrong way. In the DFA family the norm falls as K rises (Spearman −0.36, 95% interval [−0.56, −0.16]). Once label entropy and sensitivity are taken out, nothing is left (partial 0.02 [−0.21, 0.21]). In the DSL family the link is weak and positive (0.19 [0.01, 0.36]), and it disappears once the same covariates are removed (−0.07 [−0.27, 0.12]).

What the norm does follow is how well the probe's head already fits a task. Across tasks, the Spearman correlation of the norm score with the fitted head's loss is −0.96 in both families. It also follows how quickly a task is learned, but with the opposite sign to the prediction: a larger norm goes with a smaller T\*. The correlation with T\* is −0.44 [−0.70, 0.07] for DFA (13 of 16 targets never reached ε\* and are treated as +∞) and −0.56 [−0.86, −0.06] for DSL.

In the DSL family the norm predicts log T\* far better than K does. In a leave-one-target-out fit on the 15 targets that reached ε\*, the error falls from 0.65 with log K alone to 0.27 with the norm added. So the norm carries information about learnability, but as an "easiness" signal, not as complexity. The pattern holds:
- for all four estimators (E0 through E3);
- for the L2 and log-sum summaries and for either transformer block alone;
- across probe seeds (Kendall's W 0.91 for DFA, 0.95 for DSL).

Label noise lowered the norm in both families rather than raising it, the opposite of Gate 5's prediction (25% noise: DFA ratio 0.59 [0.45, 0.65], DSL 0.52 [0.50, 0.59]; the gate asked for at least 1.2).

**Gate 4 (diversity coefficient) is NOT_ESTABLISHED in both families.** It ran only after the 4-hour cap was lifted, once Gates 1–3 had been seen (Deviation 3). On the coverage mixes, the diversity coefficient of interleaved batches barely follows mixture entropy H: Spearman 0.23 [−0.02, 0.45] for DFA and −0.13 [−0.37, 0.19] for DSL (the gate asked for at least 0.7 with a lower bound above 0.5). It also does not follow H more than it follows K (gap −0.08 [−0.40, 0.28] and −0.23 [−0.50, 0.11]; the gate asked for at least 0.3). Interleaved-batch diversity stays in a narrow band (DFA 0.026–0.037, DSL 0.017–0.021). Reported, not gated: computed from single-task batches instead, diversity does follow H (0.73 DFA, 0.70 DSL) and Rao's Q (0.91, 0.89). So the coefficient registers which tasks a mix holds only when its batches are not mixed.

The rescaling check passes. Rescaling the probe's parameterization leaves the ranking of tasks unchanged (Spearman with the original 0.996 for DFA, 0.999 for DSL), so the handoff fails on Gates 1–3 alone. With a randomly initialized probe, the DFA norm carries no K or T\* signal (−0.05 with K, −0.04 partial, −0.02 with T\*; point estimates only). The DSL norm still follows T\* (−0.72), so in DSL the easiness signal does not need pretraining.

So: does Task2Vec track exact K, and T\*? It does not track K. It tracks T\* only with the opposite sign: a higher norm means faster learning, through how well the head fits, not through complexity.

## Timing pilot and freeze

See PROTOCOL Section 2.8. The pilot ran on a contended machine and projected 12.9 h, so the pre-declared fallback (8 diversity batches instead of 12) was applied before the freeze (commit d22ad34e).

## Deviations (details in CKPT_task2vec_complexity.md)

1. **Wall cap and job order (results not seen).** With most of the 4-hour cap used up because of machine contention, the remaining jobs were reordered so that gate inputs ran first. The order became:
   1. all-estimator embeddings;
   2. label noise;
   3. diversity jobs holding the coverage mixes;
   4. everything else.

   The run stopped at 14,121 s with 363 of 503 jobs not run:
   - 4 DSL label-noise jobs;
   - all 224 diversity jobs;
   - 108 rescaled-probe jobs;
   - 27 random-init-probe jobs.

   The finished jobs were merged, and the 10-19 report marked analyses whose inputs were missing as not run (NOT_RUN meant "not tested", not "failed"). The run can be resumed (`run.py --output results --resume`) if the cap is raised. It was resumed on 10-20 (Deviation 3).
2. **Embedding storage.** The two families' embeddings have different dimensions (DFA 1,564, DSL 1,566), so `embeddings.npz` holds one set of arrays per family (`dfa_keys`, `dfa_f`, …). The keys are as in PROTOCOL Section 8.
3. **Wall cap lifted after results were seen (10-20).** Gates 1–3 in both families and DFA Gate 5 had been seen on 10-19. Gate 4, DSL Gate 5 and the rescaled-probe and random-init checks had not, because their jobs never ran. Gate 4 had been cut by the cap, not tested, so `config.json` `max_wall_seconds` was raised from 14,400 to 57,600 s. That is the only change: no analysis code, seed, threshold, gate or job definition changed. The run resumed with `run.py --output results --resume` on a free machine, and the 363 jobs finished in 5,042 s (cumulative wall 19,163 s, 5.3 h, over the pre-registered 4 h). Gates 1–3 and DFA Gate 5 recomputed identically and stand as reported on 10-19. Gate 4, DSL Gate 5 and the two checks are results from past the pre-registered budget.

## Verification

`verify.py` passes (results/verify.json, verify.log), rerun on 10-20 over the complete results. It reloads the 5 probes per family and re-evaluates them (max difference 0 bits). It recomputes exact K for every task and rebuilds the roster identically with disjoint roles. It recomputes 3,634 saved embeddings with max relative difference 0: a 10% random sample of task embeddings plus every Gate-input embedding (1,959), and a 10% random sample of diversity batch embeddings plus every interleaved coverage-mix batch behind Gate 4 (1,675). It recomputes 245,776 statistics, bootstrap draws and table entries with the independently written independent_stats.py (max absolute difference 8.3e-16). One verify.py bookkeeping fix was needed on 10-20 (a recomputation job holding both families' embeddings could not be stacked; rows are now padded and sliced back), with no change to what is checked (CKPT).

## Limitations

- Gate 4, DSL Gate 5 and the rescaling and random-init checks ran past the pre-registered 4 h cap, after Gates 1–3 had been seen (Deviation 3). Only the cap changed, so no choice made after seeing results enters their numbers, but the run as a whole exceeded its frozen budget.
- T\* is heavily right-censored for DFA (104 of 128 α = 1 runs never reached ε\* = 0.5 bits), so the DFA Gate 3 estimate rests on 3 finite targets plus ties.
- The probes are tiny transformers pretrained on the same family. A large pretrained probe on real data could behave differently. These results speak about this setting only.
- The negative link between the norm and head loss suggests the E0 norm mostly measures how well the frozen features fit the task. The research note's Kolmogorov reading of the norm does not hold here.

![Task2Vec versus exact K](expt_v1/results/task2vec_vs_K.png)

Rows: DFA tasks (Experiment 11), DSL tasks (Experiment 12). Columns: log L1 norm of the E0 embedding against exact K (gray: each of the 4 probes; blue: mean over probes); the same after removing label entropy and sensitivity (rank residuals); K_hat against T* on the training targets (hollow squares: never reached eps*, treated as +inf); diversity coefficient against mixture entropy on the coverage mixes (gray bars: 95% bootstrap interval over batch pairs and probes; dashed: reference bounds).

## Gates

| Gate | Statistic | Family | Estimate | 95% interval | Threshold | Family passes |
|---|---|---|---|---|---|---|
| G1 | Spearman(K_hat, K) on embedded tasks | DFA | -0.359 | [-0.555, -0.159] | ≥ 0.5, lower bound > 0.2 | no |
| G1 | Spearman(K_hat, K) on embedded tasks | DSL | 0.19 | [0.0112, 0.357] | ≥ 0.5, lower bound > 0.2 | no |
| G2 | Partial Spearman(K_hat, K \| label entropy, sensitivity) | DFA | 0.0166 | [-0.209, 0.21] | ≥ 0.3, lower bound > 0 | no |
| G2 | Partial Spearman(K_hat, K \| label entropy, sensitivity) | DSL | -0.072 | [-0.272, 0.116] | ≥ 0.3, lower bound > 0 | no |
| G3 | Spearman(K_hat, T*) on targets (alpha = 1, censored = +inf) | DFA | -0.436 | [-0.697, 0.0673] | ≥ 0.5, lower bound > 0 | no |
| G3 | Spearman(K_hat, T*) on targets (alpha = 1, censored = +inf) | DSL | -0.564 | [-0.855, -0.0572] | ≥ 0.5, lower bound > 0 | no |
| G4 | Diversity vs mixture entropy H, coverage mixes | DFA | 0.233 | [-0.0184, 0.452] | ≥ 0.7, lower bound > 0.5 | no |
| G4 | Diversity vs mixture entropy H, coverage mixes | DSL | -0.128 | [-0.368, 0.194] | ≥ 0.7, lower bound > 0.5 | no |
| G4 | rho_H - \|rho_K\|, coverage mixes | DFA | -0.081 | [-0.396, 0.275] | ≥ 0.3, lower bound > 0 | no |
| G4 | rho_H - \|rho_K\|, coverage mixes | DSL | -0.229 | [-0.505, 0.114] | ≥ 0.3, lower bound > 0 | no |
| G5 | L1(25% label noise) / L1(clean), median over tasks | DFA | 0.592 | [0.45, 0.653] | ≥ 1.2, lower bound > 1 | no |
| G5 | L1(25% label noise) / L1(clean), median over tasks | DSL | 0.517 | [0.496, 0.595] | ≥ 1.2, lower bound > 1 | no |

| Gate | Verdict (both families must pass) |
|---|---|
| G1 | **NOT_ESTABLISHED** |
| G2 | **NOT_ESTABLISHED** |
| G3 | **NOT_ESTABLISHED** |
| G4 | **NOT_ESTABLISHED** |
| G5 | **NOT_ESTABLISHED** |

Handoff to real data (Gates 1-3 supported, Kendall's W ≥ 0.8, rescaled rank correlation ≥ 0.9, non-negative incremental value in both families): **not met**.

## Estimators and summaries (reported, not gated)

| Embedding | Family | Gate 1 rho [CI] | Gate 2 partial [CI] | Gate 3 rho [CI] |
|---|---|---|---|---|
| E1 | DFA | -0.382 [-0.562, -0.175] | -0.0146 [-0.228, 0.193] | -0.401 [-0.689, 0.0843] |
| E1 | DSL | 0.188 [0.0097, 0.357] | -0.0691 [-0.27, 0.125] | -0.535 [-0.841, -0.0435] |
| E2 | DFA | -0.367 [-0.557, -0.162] | 0.00936 [-0.209, 0.209] | -0.401 [-0.706, 0.0674] |
| E2 | DSL | 0.184 [0.01, 0.355] | -0.0788 [-0.275, 0.114] | -0.564 [-0.855, -0.0616] |
| E3 | DFA | -0.283 [-0.459, -0.0693] | -0.111 [-0.283, 0.0825] | -0.574 [-0.764, -0.0627] |
| E3 | DSL | 0 [-0.18, 0.182] | -0.0923 [-0.281, 0.104] | -0.427 [-0.739, 0.156] |
| E0, rescaled probe | DFA | -0.36 [-0.554, -0.153] | 0.0172 [-0.203, 0.216] | -0.401 [-0.688, 0.0843] |
| E0, rescaled probe | DSL | 0.194 [0.014, 0.361] | -0.0673 [-0.271, 0.127] | -0.564 [-0.855, -0.0692] |
| E0, L2 norm | DFA | -0.359 [-0.553, -0.143] | -0.00239 [-0.232, 0.194] | -0.367 [-0.703, 0.166] |
| E0, L2 norm | DSL | 0.185 [0.0131, 0.356] | -0.0953 [-0.275, 0.0836] | -0.567 [-0.852, -0.0194] |
| E0, log-sum | DFA | -0.363 [-0.556, -0.156] | 0.0251 [-0.197, 0.225] | -0.401 [-0.698, 0.0327] |
| E0, log-sum | DSL | 0.169 [-0.0145, 0.342] | -0.0776 [-0.29, 0.144] | -0.605 [-0.848, -0.1] |
| E0, block 1 only | DFA | -0.371 [-0.552, -0.178] | -0.0154 [-0.229, 0.177] | -0.436 [-0.732, 0.0674] |
| E0, block 1 only | DSL | 0.138 [-0.0378, 0.318] | -0.072 [-0.273, 0.146] | -0.642 [-0.862, -0.181] |
| E0, block 2 only | DFA | -0.38 [-0.566, -0.182] | -0.0136 [-0.249, 0.189] | -0.436 [-0.713, 0.0282] |
| E0, block 2 only | DSL | 0.196 [0.0102, 0.365] | -0.0482 [-0.268, 0.163] | -0.591 [-0.844, -0.101] |
| E0, random-init probe (point only) | DFA | -0.0531 | -0.0407 | -0.0216 |
| E0, random-init probe (point only) | DSL | 0.119 | -0.0154 | -0.718 |

## Probe and parameterization sensitivity

| Family | Kendall's W | mean pairwise Spearman | probe pairs disagreeing on K-level order | Spearman(K_hat, rescaled K_hat) | Gate 1 rho change, rescaled | median L1 ratio rescaled/original |
|---|---|---|---|---|---|---|
| DFA | 0.911 | 0.882 | 0 | 0.996 | -0.00101 | 1.28 |
| DSL | 0.95 | 0.933 | 0.833 | 0.999 | 0.00408 | 1.35 |

## Mechanism

| Family | rho khat head loss | rho khat head error | rho khat label entropy | rho khat sensitivity | rho head loss K | rho head error K | rho head loss sensitivity | partial head loss K given sensitivity | partial khat K given head loss | rho sensitivity K | rho label entropy K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DFA | -0.956 | -0.949 | -0.02 | -0.649 | 0.319 | 0.325 | 0.622 | -0.0279 | -0.195 | 0.543 | -0.0239 |
| DSL | -0.961 | -0.916 | -0.706 | -0.781 | -0.167 | -0.111 | 0.723 | -0.156 | 0.107 | -0.0826 | -0.408 |

## Incremental value over exact K for T*

- DFA: partial khat tstar given K = -0.268; partial khat tstar given K entropy sensitivity = 0.12; rho K tstar = 0.498; n finite targets = 3; n censored targets = 13; loo rmse note = only 3 uncensored targets; leave-one-out RMSE not computed; rho khat code length proxy = -0.776; rho K code length proxy = 0.752
- DSL: partial khat tstar given K = -0.566; partial khat tstar given K entropy sensitivity = -0.2; rho K tstar = -0.0371; n finite targets = 15; n censored targets = 5; loo rmse logK = 0.647; loo rmse logK plus lognorm = 0.267; rho khat code length proxy = -0.555; rho K code length proxy = -0.0245

## E0 against the per-example Fisher (E2)

- DFA: median (E0 L1 × batch size) / (E2 L1) = 0.998; Spearman by probe [0.987, 0.974, 0.977, 0.983]
- DSL: median (E0 L1 × batch size) / (E2 L1) = 0.997; Spearman by probe [0.996, 0.997, 0.995, 0.995]

## Diversity coefficient (reported analyses)

| Family | Mixes | Batches | rho with H | rho with K | rho with Rao Q | other |
|---|---|---|---|---|---|---|
| DFA | coverage (40) | interleaved | 0.233 | 0.314 | 0.136 | range [0.0264, 0.0366] |
| DFA | coverage (40) | single-task | 0.729 | -0.366 | 0.913 | range [0.0269, 0.162] |
| DFA | Experiment 11/12 training mixes (64) | interleaved | 0.151 | 0.168 | 0.105 | rho with alpha -0.148; alpha = 1 slice rho with K 0.461 (n = 16) |
| DFA | reference bounds | — | — | — | — | lower 7.45e-16, upper 0.0329 |
| DSL | coverage (50) | interleaved | -0.128 | 0.101 | -0.164 | range [0.0167, 0.0213] |
| DSL | coverage (50) | single-task | 0.695 | 0.0343 | 0.892 | range [0.0167, 0.0633] |
| DSL | Experiment 11/12 training mixes (80) | interleaved | 0.194 | -0.0555 | 0.0431 | rho with alpha -0.206; alpha = 1 slice rho with K -0.123 (n = 20) |
| DSL | reference bounds | — | — | — | — | lower 0.013, upper 0.0174 |

## Label noise

- DFA: L1 ratio noisy/clean by rate {'0.10': 0.8003909414970412, '0.25': 0.5924240979683605} (16 tasks)
- DSL: L1 ratio noisy/clean by rate {'0.10': 0.7911638697171391, '0.25': 0.517080329549982} (20 tasks)

## Censoring of T*

- DFA: 104 of 128 alpha = 1 training runs never reached eps* = 0.5 bits; 13 of 16 targets have a censored median T* (ranked highest, tied).
- DSL: 35 of 160 alpha = 1 training runs never reached eps* = 1 bits; 5 of 20 targets have a censored median T* (ranked highest, tied).

## Run

Wall time 5.32 h on a shared 4-core, 15 GB CPU machine (no GPU); versions {'python': '3.12.3', 'torch': '2.14.1+cpu', 'numpy': '2.5.4', 'scipy': '1.18.1'}; bootstrap 20000 draws, seed 29101426. Full numbers: expt_v1/results/gates.json, analysis.json, task_table.csv, diversity.csv.
