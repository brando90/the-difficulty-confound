# Protocol: Floor or late learning? Experiment 20's low-share LSTM cells run 4× longer (`21_dfa_lstm_low_alpha_long_budget`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/21_dfa_lstm_low_alpha_long_budget/PROTOCOL.md>

Status: Frozen on 10-20-2026 before any Experiment 21 training. The freeze commit is logged in [CKPT_dfa_lstm_long_budget.md](CKPT_dfa_lstm_long_budget.md). After the freeze nothing in Sections 2–6 changes; deviations go in RESULTS.

This is the follow-up that [Experiment 20](../20_dfa_lstm_complexity_vs_alignment/PROTOCOL.md)'s RESULTS left open. Brando Miranda asked for these experiments to be iterated without asking ("Go", 10-18-2026). Notation is that of Experiments 11 and 20.

## 1. The question

In Experiment 20 every LSTM run at target share $\alpha \in \{0.05, 0.15\}$ ended at 0.85 to 1.0 bits of held-out excess loss at $T = 2^{16}$ examples, at every DFA size $n$, and all 256 were flagged as plateaued. Its gate G6 (an alignment floor $a_1(1-\alpha)$, $a_1 = 0.84$ bits) passed. But a floor inside the budget cannot be told apart from learning that starts later: at $\alpha = 0.15$ the target was seen only 9,830 times, interleaved with 16 other languages.

**Does the floor hold when the same runs go 4× longer, or do they learn late?**

## 2. What was seen before this protocol was written (disclosure)

- **All of Experiment 20's results**: every gate, the RESULTS tables, and in particular the mean $\varepsilon(2^{16})$ per $(n, \alpha)$ cell (for example 0.96 and 0.85 bits at $n = 4$, $\alpha = 0.05$ and 0.15; 0.33 bits at $\alpha = 0.4$; 0.0004 bits at $\alpha = 1$), the plateau census and the post hoc effective-data check.
- **The per-length label entropy of the 16 targets** (a property of the tasks, no training): the "marginals-only" loss $H(y \mid \text{task}, \ell)$ is 0.929 to 0.997 bits, mean 0.981.
- **The data stream of this design** (Section 3.2), checked with `data.assemble_mix` before the freeze.

No run of Experiment 21 exists, and no Experiment 20 run beyond $2^{16}$ exists. Every Experiment 21 number is prospective. The comparisons with Experiment 20 rows (Section 4.3) use seen data on the reference side only.

## 3. Design

### 3.1 Unchanged from Experiment 20

The 16 targets and 16 distractors from task seed 20261011, the shared 4,096-string test set and the 69,632-string pools, the exact octave mix construction, single-length batches of 64, the 84,290-parameter LSTM, AdamW at $10^{-3}$ (weight decay 0.01, betas (0.9, 0.999), clip 1.0, 8-step warmup, constant afterwards), the 8 training seeds 11101 … 11808, one thread per run, and evaluation on the target's test set at every checkpoint. `dfa.py`, `data.py` and `model.py` are byte-identical to Experiment 20's. `run.py --check-only` passed, and the regenerated `tasks.json` and all 64 `data/*.npz` files are byte-identical (SHA-256) to Experiment 20's.

### 3.2 What changes

1. **Only the low shares**: $\alpha \in \{0.05, 0.15\}$. 4 levels × 4 targets × 2 shares × 8 seeds = **256 runs**.
2. **Budget $2^{18}$ examples** (4,096 steps), checkpoints at $T = 2^9 … 2^{18}$ (10). This is the largest power-of-two budget with **no pool reuse**: at $\alpha = 0.15$ the target is drawn 39,322 times, about 2,300 per length, under the 4,096 per length in the pool. At $2^{19}$ the target pool would be reused. The target is seen 13,108 ($\alpha = 0.05$) and 39,322 ($\alpha = 0.15$) times by $2^{18}$; each distractor about 15,560 and 13,930 times.
3. **The data order is re-drawn, not extended.** `assemble_mix` draws the task of every slot octave by octave and then the length schedule and the pool order for the whole budget. So through $2^{16}$ the sequence of which task fills each slot is identical to Experiment 20's (checked for all 256 cells by `verify.py`), while the lengths and the strings differ. The $2^{16}$ checkpoint of this experiment is therefore a **replicate** of Experiment 20's cell on a re-drawn string order, not the same run. The verdicts compare $2^{16}$ with $2^{18}$ **within the same Experiment 21 run**.
4. **Cell order**: $\alpha = 0.15$ first, then $\alpha = 0.05$; within each, $n = 4, 8, 16, 32$. Cell cost does not depend on $n$, so the decisive cells finish first.
5. **Wall cap** 43,200 s (12 h) cumulative, in resume-safe chunks of at most 2 h, with done markers written last. The timing probe under a shared box gave about 300 s per cell, 5.4 h for the grid. If the cap is reached, the analysis runs on the finished cells, cells with missing runs are `INCOMPLETE`, and RESULTS lists them.
6. **New analysis** (`analyze.py`, `verify.py`), since Experiment 20's laws and gates need $\alpha \in \{0.4, 1\}$. `report.py` only formats `analysis.json` and is written after the run.

## 4. Measurements and analysis

Per run $r$ (target $j$, seed $s$, share $\alpha$): $\varepsilon_r(T)$ at the 10 checkpoints. Write $e_{16} = \varepsilon(2^{16})$ and $e_{18} = \varepsilon(2^{18})$ of the same run.

### 4.1 Per run

- drop $d_r = e_{16} - e_{18}$ (bits);
- crossed: $e_{18} \le 0.5$ bits (Experiment 20's primary $\varepsilon^\star$);
- moved: $d_r \ge 0.1$ bits;
- plateaued at the end: $e_{18} \ge 0.8\, e_{16}$ and $e_{18} > 0.1$ bits (Experiment 20's definition, two octaves back);
- $T^\star(0.5)$ by Experiment 11's first-crossing log–log rule over the 10 checkpoints, censored at $2^{18}$, and in target examples $\alpha T^\star$.

### 4.2 Per $(n, \alpha)$ cell (32 runs) and its verdict

Means over runs; 95% intervals from the seed-block bootstrap (8 seeds, 20,000 draws, analysis seed 21102026, percentile). The cell verdict, in this order:

| Verdict | Rule |
|---|---|
| `LATE_LEARNING` | lower bound of the mean drop > 0.1 bits **and** at least 8 of 32 runs crossed 0.5 bits by $2^{18}$ |
| `FLOOR_HOLDS` | upper bound of the mean drop < 0.05 bits **and** at most 1 of 32 runs crossed |
| `DESCENDING` | lower bound of the mean drop > 0.05 bits (moving, not yet learning) |
| `UNRESOLVED` | otherwise |

### 4.3 Overall verdict (the answer to Section 1)

Decided on $n \in \{4, 8, 16\}$, the sizes the LSTM learns within $2^{16}$ at $\alpha = 1$ (Experiment 20). At $n = 32$ even $\alpha = 1$ ends at 0.67 bits, so a floor there cannot be told from not learning the language at all; those two cells are reported, not decisive.

- **FLOOR**: all six decisive cells are `FLOOR_HOLDS`. The Experiment 20 floor survives a 4× budget.
- **LATE_LEARNING**: the cells $(n = 4, \alpha = 0.15)$ and $(n = 8, \alpha = 0.15)$ are both `LATE_LEARNING`. These have the most target data and the smallest automata, so if the floor is a budget artifact they break first. The floor was then, at least there, learning that had not started yet.
- **MIXED**: anything else, reported cell by cell.

### 4.4 Reported, not part of the verdict

- **Interference at matched target examples** (prospective on the Experiment 21 side): $I_r = \log_2[\max(e_{18}, 10^{-3}) / \varepsilon^{(\alpha = 1)}_{j,s}(\alpha \cdot 2^{18})]$, where the reference is Experiment 20's $\alpha = 1$ run of the same target and seed, interpolated log–log at 13,107 or 39,322 examples (floored at $10^{-3}$). $I > 0$ means the mixed run is worse than a pure run that saw as many target examples: the distractors interfere. Also at $2^{16}$ ($I$ at 3,277 and 9,830 examples). Cell mean with seed-bootstrap interval, and median.
- **Replicate check**: $e_{16}$ here minus Experiment 20's $\varepsilon(2^{16})$ for the same cell (re-drawn string order), mean with interval.
- **Marginals-only reference**: $e_{18} - H(y \mid \text{task}, \ell)$ per cell. A run at about 0 has learned only the per-length base rate.
- **Distractor losses**: the mean held-out excess loss over the 16 distractors at $2^{16}$ and $2^{18}$, from the saved checkpoints (`analyze.py --distractors`). This shows whether the network learns the other languages while the target stays flat.
- Mean curves by cell, the plateau census at $2^{18}$, and $T^\star$ geometric means with censoring counts.

## 5. Predictions (written before any run; scored in RESULTS, not gates)

- **Overall LATE_LEARNING** (about 60% confidence). Experiment 20's $\alpha = 0.4$ runs reached 0.33 bits at $n = 4$ with 26,214 target examples; here $\alpha = 0.15$ sees 39,322 by $2^{18}$.
- $(4, 0.15)$ and $(8, 0.15)$ `LATE_LEARNING`; $(16, 0.15)$ `DESCENDING` or `LATE_LEARNING`.
- $\alpha = 0.05$: $(4, 0.05)$ `DESCENDING` or `LATE_LEARNING`; $(8, 0.05)$ and $(16, 0.05)$ `FLOOR_HOLDS` or `DESCENDING`.
- $n = 32$: `FLOOR_HOLDS` at both shares.
- Interference $I > 0$ in every decisive cell at $2^{18}$ (the mixed runs are worse than pure runs with equal target data).
- The replicate check is within ±0.05 bits of Experiment 20 on average.

## 6. Artifacts and verification

Under `expt_v1/results/`: `tasks.json`, `data/`, `mixes.json`, `metrics.csv` (2,560 rows), `cells/`, `predictions/`, `checkpoints/` (out of git, hashed), `run_state.json`, `manifest.json`, `distractors.json`, `analysis.json`, `verification.json`.

`verify.py` must report PASS before anything is written up: (1) training sources match the manifest; (2) tasks regenerate identically with Hopcroft = Moore, and the data regenerate identically and are byte-identical to Experiment 20's; (3) all 256 mixes regenerate, none reuses the pool, and all 256 task-slot sequences through $2^{16}$ equal Experiment 20's; (4) every checkpoint replays to max $|\Delta| < 10^{-9}$ in bits, accuracy and per-string predictions; (5) the first cell retrains from scratch to the same metrics; (6) `metrics.csv` equals the cell records; (7) `analysis.json` recomputes identically and the cell means, drops, crossing counts and verdicts agree with an independent recomputation; (8) 8 cells' distractor losses replay identically.

## 7. Scope and limits

- One learner, one model size, one learning rate (constant after warmup), lengths 8–24, unrelated distractors, as in Experiment 20. A 4× budget can show late learning; FLOOR here means "no learning by $2^{18}$", not a proof of an asymptotic floor.
- The constant learning rate is not retuned for a longer run. A schedule with decay could change when learning starts.

**TLDR-end:** [tdc: Expt 21 design] Experiment 20's 256 LSTM runs at target share 0.05 and 0.15, which all stalled near 0.85–1.0 bits at 2^16 examples, rerun to 2^18 with the same tasks, data rule, learner and seeds; a pre-registered within-run verdict (late learning, floor, descending) decides whether Experiment 20's alignment floor is real or a budget artifact.
