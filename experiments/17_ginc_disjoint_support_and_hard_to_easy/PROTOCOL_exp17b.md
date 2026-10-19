# Protocol: Experiment 17b, coverage-matched easy pools for Experiment 17's arm H (hard → easy)

Status: FROZEN 10-19-2026, before any gated run. The freeze commit fixes this file, `expt_v2_coverage/config_v2.json`, `run_v2.py`, `analyze_v2.py`, `report_v2.py`, `verify_v2.py` and the concept list in `expt_v2_coverage/results_check_v2.json`. Deviations go in `RESULTS_exp17b.md`.

## 1. Why

Experiment 17's surprise: zero-shot, training only on hard (n = 32) concepts left 2.75 bits/token of excess on new easy (n = 4) targets, against 6.71 after training only on easy concepts. The post-hoc reading (RESULTS_exp17.md, Takeaway 2) was token coverage, not difficulty:

- each easy-pool concept emits only 11–15 of the 32 tokens (at most 16 are possible with 4 states × 4 emission tokens), while each hard-pool concept emits 31–32;
- the easy pool as a whole never emits token 18, which easy target `hmm_n004_t1` uses.

Question: **once the easy pool's token coverage matches the hard pool's, does hard-only training still beat easy-only training zero-shot on easy targets?**

## 2. New pools (everything else is Experiment 17's)

Both pools hold 8 n = 4 concepts on tokens 0–31, pass Experiment 17's checks 1–2 (strongly connected and aperiodic, process rank = n = 4) and its entropy band around the same h_ref = 2.809 bits (bands 0.15, 0.30, 0.50 tried in order, one band per pool). Each concept's states emit disjoint token sets, so each token reveals its state and the entropy rate is exact (Σ π_i (H(A_i) + H(O_i))); the band uses that exact value instead of Experiment 17's simulated quick estimate. Successors, transition and emission weights are drawn as in Experiment 17 (3 successors, Dirichlet(1), q = 6 quantization; every listed token has probability ≥ 1/64 in its state). Candidates come from their own PCG64 streams (task_sampling_seed, 1717, pool code, slot, i); the first candidate in generation order that passes is accepted.

- **cov32 (primary).** 8 emission tokens per state, the 4 states a random partition of all 32 tokens: *every concept emits all 32 tokens*, as the hard pool's concepts do. K_bits = 400 (Experiment 17's description length with 8 emission tokens per state; 224 for the easy family, 2,080 for n = 32). Certified state count is still 4.
- **cov16 (secondary).** Experiment 17's exact family (4 emission tokens per state, K_bits = 224), 16 distinct tokens per concept, in complementary pairs (each pair splits a random permutation of the 32 tokens), so every token is emitted by exactly 4 of the 8 concepts.

Check-only result, fixed at the freeze: cov32 needed the 0.50 band (entropy rates 3.02–3.29 bits; with 8 tokens per state, Dirichlet(1) concepts sit near 3.9 bits, and 0 of 1,500 slot-0 candidates fall within 0.15); cov16 fits the 0.15 band (2.68–2.94). Coverage (stationary token distribution; perplexity = 2^entropy):

| Pool | distinct tokens per concept | pool union | per-concept token perplexity | pool token perplexity |
|---|---|---|---|---|
| easy (Exp 17) | 11–15 | 31 (no token 18) | 9.3 | 25.2 |
| cov16 | 16 | 32 | 10.9 | 29.4 |
| cov32 | 32 | 32 | 12.8 | 26.4 |
| hard (Exp 17) | 31–32 | 32 | 23.9 | 30.8 |

So both new pools fix the union (every token seen, at a pool share ≥ 0.7%). cov32 matches the hard pool's nominal per-concept support. Its effective per-concept breadth (12.8) is still about half the hard pool's (23.9), because the entropy band keeps its emissions peaked.

**Reused, not re-run:** the hard, easy, near (α ∈ {0, 0.05}) and α = 1 runs of Experiment 17 (`expt_v1/results`). The task list keeps concept ids 0–37 byte-identical and gives the new concepts ids 38–53. Training streams are seeded by (seed, concept id, document index), and the pool order by (seed, 7777) over 8 concepts, so the reused runs are exactly what a re-run would produce. `verify_v2.py` replays one reused hard run with the extended task list to prove it.

## 3. The grid

Per seed (17101, 17202, 17303) and per new pool: one α = 0 run scored on all 4 targets, plus one α = 0.05 run per target. 10 runs per seed, **30 gated runs**, T checkpoints 2^16 … 2^20, Experiment 17's model, optimizer, batch, α schedule and matched compute. 3 workers, one thread each, under `scratch/at343/CPU_LOCK_AT-453`.

## 4. Predictions (written before any gated run)

- If Experiment 17's zero-shot gap was coverage, the coverage-matched pools close most of it. Prediction: cov32 closes at least half of the easy-target gap at α = 0 (gap closed = (ε_easy − ε_cov32)/(ε_easy − ε_hard) ≥ 0.5), and so does cov16. Disclosed: before the freeze we looked at Experiment 17's zero-shot excess per easy target (easy pool 5.04 bits/token on `hmm_n004_t1`, which uses the never-seen token 18, and 8.39 on `hmm_n004_t2`, which does not; hard pool 2.68 and 2.82). So the missing token is not the main cause, and per-concept breadth is the candidate the new pools test.
- On the primary question our prior is about a coin flip: hard-only may keep a small zero-shot edge because its concepts' effective breadth is still twice cov32's, or the matched pool may win as Experiment 17 originally predicted.
- With 5% target data (α = 0.05), pool choice barely matters on easy targets (as in Experiment 17). On hard targets the cheap n = 4 pools beat the hard pool at T = 2^20, as the easy pool did.

## 5. Gates (fixed now)

Units: (seed, target) pairs, 3 seeds × 2 targets = 6 per contrast. Two-stage bootstrap (seeds, then targets within seed), 20,000 draws, analysis seed 29101817, Experiment 17's `two_stage` and `sign_verdict`.

- **Contrast**, for P ∈ {cov32, cov16} and α ∈ {0, 0.05}, at T_max: on easy targets Δ = ε(hard) − ε(P); on hard targets Δ = ε(P) − ε(hard). Positive = the matched pool wins. Verdict MATCHED_POOL_BETTER if mean > 0, 95% lower bound > 0 and ≥ 5/6 pairs > 0; MISMATCHED_POOL_BETTER if the mirror holds; else NOT_ESTABLISHED.
- **Primary answer (Q):** "Does hard-only still beat easy-only zero-shot on easy targets once coverage is matched?" **YES** if the cov32, α = 0, easy-target contrast is MISMATCHED_POOL_BETTER. **NO, the matched pool wins** if it is MATCHED_POOL_BETTER. Otherwise **NO RELIABLE DIFFERENCE** (NOT_ESTABLISHED). The same answer is reported for cov16 as secondary.
- **H1′ (Experiment 17's H1 with the coverage-matched pool), per pool:** SUPPORTED if the easy-target contrast is MATCHED_POOL_BETTER at both α. REFUTED if MISMATCHED_POOL_BETTER at both. Else NOT_ESTABLISHED. **H2′ crossover:** SUPPORTED if H1′ is SUPPORTED and the hard-target contrast is MATCHED_POOL_BETTER at both α. REFUTED if H1′ is REFUTED. Else NOT_ESTABLISHED.
- **Reported, not gated:** gap closed at α = 0 on easy targets (ratio of mean differences, two-stage bootstrap interval); ε(easy) − ε(P) on easy targets; per-target cell means at α = 0 (for `hmm_n004_t1`, which uses token 18, and `hmm_n004_t2`, which does not); Δ at every checkpoint; each pool's worth in target-only tokens; training-pool excess at T_max (every run learned its pool).

## 6. Verification

`expt_v2_coverage/verify_v2.py`: (a) all 30 runs present, finite, at the frozen checkpoints, α schedules exact, metrics.csv / pool_loss.csv / mixes.json rebuilt byte for byte; (b) analysis_v2.json reproduced from the two metrics files; (c) source hashes match the manifest, Experiment 17's metrics.csv matches its recorded sha256, and concept ids 0–37 equal Experiment 17's task list; (d) one cov32 run and one reused Experiment 17 hard run replayed to the first checkpoint within 1e-6 bits; (e) RESULTS tables regenerated and found in `RESULTS_exp17b.md`.

## 7. Limits stated up front

Same as Experiment 17: synthetic HMMs, 354k parameters, at most 2^20 tokens, three seeds. Coverage is matched in support, not in effective breadth or entropy: cov32 runs 0.2–0.5 bits/token above h_ref and its token distribution is half as broad as the hard pool's. The disjoint emission sets make the new concepts' states visible from their tokens, which the targets' states are not.
