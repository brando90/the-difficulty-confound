# Protocol: Experiment 23, Experiment 13-small and Experiment 17b's coverage arms trained to convergence with 8 seeds (`23_exp13_full_training_8seeds`)

Status: FROZEN 10-20-2026, before any job. The freeze commit contains only this file. Code, tests, the smoke and the
FarmShare package follow in later commits and implement exactly what is written here. Any change after the freeze is a
deviation and is listed in `RESULTS.md`.

## 1. Why

Experiment 13-small (`13_ginc_hmm_complexity_vs_alignment`, arm small) and Experiment 17b
(`17_ginc_disjoint_support_and_hard_to_easy`, `PROTOCOL_exp17b.md`) both trained the 354,240-parameter transformer for
at most 2^20 tokens on 3-4 seeds. That budget left the models partly trained: at 2^20 tokens and α = 1 the n = 32
targets were still 0.65 bits/token above the exact oracle, and Experiment 13's pilot rule had to raise the gated
excess ε* from 0.05 to 1.10 bits/token (38% of the targets' entropy rate). Experiment 17b's gates came out
NOT_ESTABLISHED with 6 (seed, target) pairs per contrast.

This experiment asks the same questions with the same concepts, documents, references, model, optimizer and α grid, but
(a) trains every run until a pre-registered stopping rule says the grid has converged, and (b) uses 8 seeds.

## 2. What is held fixed (copied, not rebuilt)

- **Concepts and references.** Experiment 13-small's task list (8 targets n ∈ {4, 8, 16, 32} × 2, 8 distractors,
  certified state counts), 512 held-out eval documents per target, exact float64 oracle and Bayes-mixture code
  lengths; Experiment 17b's task list (concept ids 0-53: targets n ∈ {4, 32} × 2, pools easy, hard, near, far, cov32,
  cov16), its eval documents and references. They are copied byte for byte from the source experiments' frozen builds
  into `prep/`, whose sha256s are committed in `prep/prep_manifest.json` and checked before any training.
- **Model.** 3 layers, d_model 96, 4 heads, MLP 384, vocabulary 64 + BOS, 128 learned positions, tied output
  embedding: 354,240 parameters (335,712 non-embedding), asserted at build. Transformers only.
- **Optimizer and data.** AdamW, lr 1e-3, betas (0.9, 0.95), weight decay 0.1 on matrices, gradient clip 1.0, 32 linear
  warm-up steps then constant learning rate; batch 8 documents × 128 tokens = 1,024 tokens per step; float32 training,
  one thread per run. Training streams exactly as in the sources: Experiment 13's Bresenham α schedule with the
  distractors cycled in a (seed, 7777)-permuted order; Experiment 17's `stream_ids` for pools. Each document is drawn
  from PCG64 seeded by (seed, concept id, document index), so a stream's first 2^20 tokens are the source run's stream
  and the extension continues it.
- **Measurement.** Exact float64 code lengths of the model on each scored target's 512 eval documents at each
  checkpoint; excess ε = model − oracle, identification ε_id = mixture − oracle, learning ε_learn = ε − ε_id, over the
  slices early (1-16), middle (17-64), late (65-128) and all; for α = 0 the target is unseen and ε_id = 0
  (Experiment 17's rule). Side losses (Experiment 13's distractors, Experiment 17's training pool) are logged, not gated.
- **Code.** `run23.py` imports Experiment 17's `hmm.py` (Experiment 13's plus two pool helpers; equality on every
  Experiment 13 function is tested), `model.py` (identical in both) and `run.py` unchanged. The analysis calls
  Experiment 13's `analyze.py` stage 1 and stage 2 unchanged, and Experiment 17b's `analyze_v2.py` logic
  parametrized only by seeds, checkpoints, pools and the pair threshold (tested to reproduce `analysis_v2.json`
  exactly on the original inputs).

## 3. Training to convergence: the stopping rule

Every run trains in stages ending at T_0 = 2^20, T_1 = 2^22, T_2 = 2^24 and T_3 = 2^26 tokens (the cap), with the
constant learning rate throughout. At each stage end the full training state (weights, AdamW moments, scheduler) is
saved and the next stage resumes it exactly (bitwise equality with an uninterrupted run is tested). Checkpoints
(evaluation and saved weights) are at every doubling 2^16, 2^17, …, up to the stage end.

**Validation set (used only by the stopping rule).** For every target, 256 documents; for every concept of the
Experiment 17b training pools, 64 documents; drawn with stream seed eval_seed + 11 = 20261031, which no eval or
training stream uses (disjointness tested), with exact oracle code lengths. A run's *monitor* at checkpoint T is its
mean validation excess over all positions: on its target for α > 0 runs; averaged over its training pool's 8 concepts
for Experiment 17b's α = 0 runs (whose target is unseen by design).

**Cells.** Experiment 13: (n, α), 16 cells, each the mean over 8 seeds × 2 targets. Experiment 17b: (pool, n) at
α = 0.05, 8 cells of 8 seeds × 2 targets, and (pool, α = 0), 4 cells of 8 seeds.

**Rule.** After stage k (all runs of the experiment finished), for every cell let before = its mean monitor at
T_k / 4 and after = at T_k. The cell has converged if before − after ≤ max(0.02 bits, 0.05 × after). If every cell
of the experiment has converged, the experiment STOPS at T_stop = T_k; otherwise all its runs continue to stage k+1.
After stage 3 it stops at the cap (T_stop = 2^26, reported as "cap", not as converged). The two experiments decide
separately. The decision table (every cell's before, after, improvement, threshold) is written to `decisions/` and
reported.

**Cooldown branch (reported, not gated).** At each stage end a copy of the model continues the same stream for
T_k / 8 more tokens with the learning rate decayed linearly from its constant value to 0, is evaluated once and
discarded. It shows how much of the remaining excess is learning-rate noise. Gates never use it.

## 4. Grid

- **Experiment 13 (e13).** Seeds 13101, 13202, 13303, 13404 (Experiment 13-small's four) and 13505, 13606, 13707,
  13808; 8 targets × α ∈ {0.05, 0.15, 0.4, 1.0}: **256 runs.**
- **Experiment 17b (e17b).** Seeds 17101, 17202, 17303 (Experiment 17's three) and 17404, 17505, 17606, 17707, 17808;
  pools cov32 (primary), cov16, hard and easy; per seed and pool one α = 0 run scored on all 4 targets and one
  α = 0.05 run per target: **160 runs.** The hard and easy pools are re-run here (not reused) because every run must
  reach the same T_stop. Experiment 17's near pool and α = 1 target-only runs are not part of Experiment 17b's gates
  and are not run, so Experiment 17b's reported "worth in target-only tokens" (t_equiv) is dropped.

Through 2^20 tokens the runs of the source seeds are the source runs (exact replay of the first checkpoint is tested;
the full overlap is compared in verification).

## 5. Hypotheses, analysis and gates

Everything is evaluated at the experiment's own T_stop: checkpoints 2^16 … T_stop.

### 5.1 Experiment 13: G1-G3 (primary), G4-G5 (reported with the same code)

Experiment 13-small's frozen analysis (`expt_v1/analyze.py`, PROTOCOL Section 5 as aligned at its freeze), unchanged:
per-seed law fits (M0_scale, M_align_eff, M_align_add, M_K_coef, M_K_exp, M_K_both) on fit rows, frozen forecasts
(sha256 recorded in stage 1 before any held-out outcome is read), held-out rows = level n = 16, α = 0.15 and
T = T_stop; seed-block bootstrap over the 8 seeds, 20,000 draws, analysis seed 29101326; Experiment 13's thresholds.

- **G1 (complexity is real):** M_K at α = 1 from model-free T* (n = 32 over n = 4, geometric means over targets) ≥ 2.0
  with seed-block and two-stage 95% lower bounds > 1.
- **G2 (complexity adds held-out value beyond alignment):** the K law (M_K_coef or M_K_exp, lower pooled fit RMSE)
  cuts held-out RMSE against M_align_eff by ≥ 10% with 95% lower bound > 0%, also on the n = 16 rows alone, with 0
  invalid forecasts.
- **G3 (coefficient or exponent):** G3a / G3b exactly as Experiment 13 (κ and γ₁ intervals in M_K_both, M_K_coef vs
  M_K_exp held-out RMSE).
- G4 (elasticity ratio E) and G5 (position split) are computed by the same code and reported.

**ε\* rule (replaces Experiment 13's pilot rule; uses fit rows only).** ε* is the smallest value on the ladder
0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.10 bits/token that at least 75% (12 of 16) of the n = 32, α = 0.4 runs reach in
ε_learn at the largest fit checkpoint T_stop / 2 (n = 32 and α = 0.4 are fit coordinates). If none reach 1.10, ε* is
1.10 and this is reported. (Applied to Experiment 13-small's own data this rule gives 1.10, as its pilot rule did.)
Gates are also computed at ε* = 1.10 for direct comparison with Experiment 13-small (reported, not gated).
The ε curve (reported) uses 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.1.

### 5.2 Experiment 17b: Q, H1′, H2′ (primary)

Experiment 17b's frozen gates (PROTOCOL_exp17b.md Section 5), with 8 seeds. Units: (seed, target) pairs, 8 seeds × 2
targets = **16 per contrast**. Two-stage bootstrap (seeds, then targets within seed), 20,000 draws, analysis seed
29101817, Experiment 17's `two_stage` and `sign_verdict`, random draws in Experiment 17b's order.

- **Contrast**, for P ∈ {cov32, cov16} and α ∈ {0, 0.05}, at T_stop: on easy targets Δ = ε(hard) − ε(P), on hard
  targets Δ = ε(P) − ε(hard). MATCHED_POOL_BETTER if mean > 0, 95% lower bound > 0 and **≥ 14 of 16** pairs > 0
  (Experiment 17b's ≥ 5/6 scaled to 16 pairs); MISMATCHED_POOL_BETTER for the mirror; else NOT_ESTABLISHED.
- **Q:** YES (hard-only still better) if the cov32, α = 0, easy-target contrast is MISMATCHED_POOL_BETTER; NO (matched
  pool better) if MATCHED_POOL_BETTER; else NO RELIABLE DIFFERENCE. cov16 secondary.
- **H1′:** SUPPORTED if the easy-target contrast is MATCHED_POOL_BETTER at both α; REFUTED if MISMATCHED at both; else
  NOT_ESTABLISHED. **H2′:** SUPPORTED if H1′ is SUPPORTED and the hard-target contrast is MATCHED_POOL_BETTER at both
  α; REFUTED if H1′ is REFUTED; else NOT_ESTABLISHED.
- Reported: gap closed (easy targets, α = 0), ε(easy) − ε(P), per-target cell means, Δ at every checkpoint,
  training-pool excess at T_stop.

### 5.3 Also reported (not gated)

The stopping decisions and T_stop per experiment; the cooldown branch's excess next to the constant-LR excess at each
stage end; the comparison with Experiment 13-small and Experiment 17b at 2^20 (identical runs for the source seeds)
and at T_stop.

## 6. Predictions (written before any job)

- The n = 32 targets keep improving well past 2^20; we expect the grids to stop at 2^24 or hit the cap, and ε* to fall
  to 0.1-0.3 bits/token.
- G1 stays SUPPORTED. G2 and G3 are open: with converged curves the K law has more signal, but M_align_eff with the
  exact offset may still forecast the held-out level.
- On Experiment 17b, longer training should shrink the zero-shot gap between pools; whether the coverage-matched pool
  then beats the hard pool reliably (Q = NO) is about a coin flip.

## 7. Compute and cluster use

FarmShare (Stanford, unsponsored research) CPU nodes, partition `normal`, QOS limits 1024 submitted / 128 running jobs.
Each array task packs 4 runs (4 processes, one thread each, `-c 4`, 6 GB): e13 = 64 tasks per stage (throttle %24),
e17b = 40 tasks per stage (%16), so at most 40 tasks (160 cores) run at once. Stage k+1 arrays depend on the decide job
of stage k; tasks exit at once if the experiment has stopped. About 425 jobs are submitted in total. Mail:
`--mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com`, one mail per array (never
ARRAY_TASKS); decide jobs mail on FAIL only. No compute on login nodes. The measured throughput of the CPU smoke and
the resulting estimate are in `README.md`; the worst case (cap at 2^26) is about 1.16 × 2^26 tokens per run.

## 8. Verification

`verify23.py`: (a) every run present at every stage up to T_stop, finite, checkpoints complete; (b) metrics files
rebuilt identically from the per-run records; (c) a sample of saved checkpoints reloaded and re-evaluated equal to the
recorded code lengths (within 1e-9 bits); (d) the analyses rerun to identical gates; (e) prep sha256s and source-code
sha256s match the manifest; (f) the source seeds' rows at T ≤ 2^20 compared with Experiment 13-small's and
Experiment 17b's committed metrics (reported; exact replay of the first checkpoint is a CPU test).

Tests before any job (`tests/`): hmm superset, stopping rule, grid packing, validation disjointness, exact replay of
an Experiment 13-small run and two Experiment 17/17b runs at 2^16, bitwise resume across stages, a miniature grid end
to end (stage, decide, aggregate), decide halting on missing runs, and the analysis wrappers reproducing Experiment
17b's `analysis_v2.json` and Experiment 13-small's `gates.json`.

## 9. Limits

Synthetic HMM concepts and one 354k-parameter model. "Converged" means the stopping rule's tolerance under a
constant learning rate, not zero excess; the cooldown branch measures the learning-rate share. The cap may bind for the
hardest cells. Experiment 17b's coverage pools match token support, not breadth (its own Section 7). Seeds overlap the
source experiments, so the 8-seed results are not independent replications of them.
