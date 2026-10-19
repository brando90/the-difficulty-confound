# Protocol: Experiment 17, hard-to-easy transfer and disjoint support on GINC-style HMMs

Status: FROZEN 10-18-2026 (pilot done: T checkpoints 2^16 … 2^20, ε\* = 1.13 bits/token by the rule in Section 3.3, from the n = 32 pilot's learning term 1.128 at near α = 0.4, T = 2^19). The freeze commit fixes this file, `expt_v1/config.json` (T range and ε* set by the pilot rules below) and the code before any gated run. Deviations go in RESULTS and `CKPT_exp17.md`.

## 1. Why

Experiment 13-small found that complexity moves the data bill more than alignment does (elasticity ratio E = e_α/e_K = 0.80 < 1). Its alignment axis was narrow, though. The target was always in the training mix at share α, and the rest of the mix was other HMMs over the *same* 64 tokens. Brando put two sharper versions of the question (WhatsApp, 10-18-2026):

1. **Putnam vs GSM8K (same domain, different difficulty), the primary arm H.** Does training only on *hard* problems buy the *easy* ones? If difficulty were a total order (hard ⊇ easy), training on hard data should transfer to easy targets at least as well as training on easy data does. If it is not, the matched (easy) data should win on easy targets even though it is "simpler".
2. **Putnam vs poetry (disjoint support), the secondary arm D.** Does data from a domain that shares nothing with the target buy anything? And under such genuine mismatch, does alignment finally dominate complexity (E > 1)?

## 2. Setup (shared, reused from Experiment 13-small)

- **Concepts.** Experiment 13's sparse single-chain HMMs: n states, 3 successors and 4 emission tokens per state, probabilities quantized to q = 6 bits. Each concept is accepted only if it is strongly connected and aperiodic, its exact process rank equals n (certified minimal state count), and its entropy rate is within a band of the median calibration value (bands tried in order 0.15, 0.30, 0.50 bits). Complexity K is the description length in bits, K_bits(n) = n(3⌈log2 n⌉ + 4⌈log2 32⌉ + 5q): 224 bits at n = 4 and 2,080 at n = 32.
- **Tokens.** Vocabulary 64 (+ BOS). Region A = tokens 0–31, region B = 32–63. Every target, pool and near concept emits only region-A tokens.
- **Targets.** 2 at n = 4 ("easy", GSM8K) and 2 at n = 32 ("hard", Putnam), never in any pool.
- **Pools** (8 concepts each, cycled in a seed-permuted order):
  - `easy`: 8 fresh n = 4 concepts (region A).
  - `hard`: 8 fresh n = 32 concepts (region A).
  - `near`: the mixed control, 2 each at n = 4, 8, 16, 32 (region A), as in Experiment 13.
  - `far`: the *same 8 near concepts* with every token moved by +32 (region B). Same transitions, emission weights, process rank and entropy rate. Only the support differs.
- **Model and training.** Experiment 13-small exactly: 3-layer transformer, d = 96, 4 heads, MLP 384, 354,240 parameters; AdamW lr 1e-3, betas (0.9, 0.95), weight decay 0.1, grad clip 1, 32 warmup steps, batch 8 documents of 128 tokens, one thread per run.
- **α schedule.** Experiment 13's exact Bresenham rule: after m documents, exactly ⌊mα⌋ are target documents; the rest come from the pool. α = 0 means the target is never seen (zero-shot), α = 1 means target only (pool `none`).
- **Matched compute.** Every run trains on the same number of tokens with the same model and optimizer. Only the pool and α change.
- **Measurement.** At each checkpoint T, float64 code lengths on 512 held-out documents of the target. Excess ε = model − oracle (bits/token, all 128 positions; early/middle/late slices also recorded). ε_id = Bayes-mixture − oracle, exact (zero at α ∈ {0, 1}). Evaluation documents never appear in any training stream (checked).

## 3. The grid

Per seed (3 seeds: 17101, 17202, 17303), 32 runs:

| Arm | Pool | α | Targets | Runs |
|---|---|---|---|---|
| anchor | none | 1 | each of 4 | 4 |
| H | easy, hard | 0.05 | each of 4 | 8 |
| H, D | easy, hard, near, far | 0 | one run, scored on all 4 targets | 4 |
| H control, D | near | 0.05, 0.4 | each of 4 | 8 |
| D | far | 0.05, 0.4 | each of 4 | 8 |

96 gated runs. Arm H's runs (plus `near` at α ≤ 0.05 and the α = 1 anchor) run first.

### 3.3 Pilot (disclosed, seed 17909, two pilot concepts at n = 4 and 32, not in any pool)

Runs: α = 1; near α = 0.4; near α = 0.05; hard α = 0.05 for each pilot concept, plus hard α = 0 (9 runs, T to 2^20). Rules, fixed now:

- **T range.** Primary checkpoints 2^16 … 2^20 (5, doubling, Experiment 13-small's range). If the projected grid wall time at the measured throughput exceeds 4.5 h, use the fallback 2^16 … 2^19. *Set at the freeze from the pilot:* the pilot ran with 2^14 … 2^20; at 2^14 and 2^15 (16 and 32 optimizer steps, inside the 32-step warmup) the excess was 40–57 bits/token, the initialization transient of the tied N(0, 1) embeddings, not learning on any power law. So the gated range starts at 2^16, as in Experiment 13. Projected grid time was 3.8 h at 7 checkpoints under a background load of about 10, below 4.5 h, so the primary range is used.
- **ε\*** (used by arm D's elasticities only; arm H's gates do not use it). Experiment 13's rule: start at 0.05 bits/token; if the n = 32 pilot's learning term at near α = 0.4 and the second-to-last checkpoint exceeds 2 × 0.05, ε\* = that value rounded up to 0.01.

## 4. Predictions (written before any gated run)

**Arm H (primary).** Hypothesis under test: *difficulty is not a total order*. Prediction: on easy targets, training on the easy pool leaves less excess than training on the hard pool at the same compute, both zero-shot (α = 0) and with a little target data (α = 0.05). Hard-only training does not subsume the easy end. Mirror prediction: on hard targets the hard pool beats the easy pool (matched data wins both ways, a crossover). The mixed pool lands between the two on each target. The competing prediction, "hard ⊇ easy" (a total order), is that the hard pool does at least as well as the easy pool on easy targets.

**Arm D (secondary).** Prediction: (D1) zero-shot, far data buys essentially nothing on the target (excess far ≫ near). (D2) Under genuine mismatch alignment dominates: E_far > 1. (D3) E_far exceeds E_near by at least 0.10. Our honest prior on D2 is close to a coin flip. If far data contributes nothing, alignment acts as pure effective data (η → 1), which would lift Experiment 13's 0.80 only to about 1.0. E_far ≫ 1 would need the far data to actively hurt.

## 5. Gates (fixed now)

Units for arm H: (seed, target) pairs, 3 seeds × 2 targets = 6 per contrast. Intervals: two-stage bootstrap (seeds, then targets within seed), 20,000 draws, analysis seed 29101817.

- **H contrast** at α ∈ {0, 0.05}: Δ = ε(mismatched pool) − ε(matched pool) at T_max, bits/token. For easy targets the mismatched pool is `hard`; for hard targets it is `easy`. Verdict per contrast: MATCHED_POOL_BETTER if mean > 0, 95% lower bound > 0 and ≥ 5 of 6 pairs > 0; MISMATCHED_POOL_BETTER if mean < 0, upper bound < 0 and ≥ 5 of 6 pairs < 0; else NOT_ESTABLISHED.
- **H1, hard does not subsume easy (primary gate):** SUPPORTED if the easy-target contrast is MATCHED_POOL_BETTER at both α = 0 and α = 0.05. REFUTED (consistent with a total order) if MISMATCHED_POOL_BETTER at both. Otherwise NOT_ESTABLISHED.
- **H2, crossover:** SUPPORTED if H1 is SUPPORTED and the hard-target contrast is MATCHED_POOL_BETTER at both α. REFUTED if H1 is REFUTED. Otherwise NOT_ESTABLISHED.
- **Reported, not gated:** Δ at every checkpoint, the log2 ratio of excesses, the mixed pool's mean excess, and each pool's worth in target-only tokens (the α = 1 curve of the same seed and target, log-log interpolated; "<" or ">" when outside the curve).
- **D1 zero-shot:** Δ = ε(far, α = 0) − ε(near, α = 0) at T_max over 12 (seed, target) pairs. SUPPORTED if mean > 0, lower bound > 0 and ≥ 10 of 12 positive.
- **D2, E_far > 1:** fit Experiment 13's M_K_both law (K in bits, ε_id offset, censored residual at ε_res = 0.015) per seed on far rows (α ∈ {0.05, 0.4}) plus the shared α = 1 rows, all checkpoints. E = mean_s e_α / mean_s e_K at ε\*, K = K_ref (the geometric mean of 224 and 2,080 bits), as in Experiment 13. SUPPORTED if E_far > 1, its seed-bootstrap lower bound > 1 and every per-seed E > 1. If all of these fall below 1, the verdict is "NOT_ESTABLISHED (reverse)". Otherwise NOT_ESTABLISHED. The same fit is run on near rows for E_near.
- **D3:** E_far − E_near ≥ 0.10 with every per-seed difference > 0 → SUPPORTED. The mirror case is REVERSED. Otherwise NOT_ESTABLISHED.
- **Reported, not gated:** η from M_align_eff per arm, E across ε ∈ {0.05, 0.1, 0.2, 0.5, 1.0}, the model-free E from interpolated T\* (α 0.05 vs 1, K 4 vs 32), and far − near excess at α = 0.05 and 0.4.

## 6. Verification

`verify.py` (a) rebuilds metrics.csv from the per-run JSON files and checks it byte for byte, (b) re-runs the analysis and compares every number in analysis.json, (c) checks source hashes against the manifest, (d) replays two gated runs to their first checkpoint and matches the recorded excess to 1e-6 bits, (e) regenerates RESULTS tables and checks them against the committed file.

## 7. Limits stated up front

Synthetic HMMs, a 354k-parameter model and 2^20 tokens (at most) are far from Putnam and GSM8K. "Easy" and "hard" here differ in certified state count only. Three seeds give coarse intervals. A real-LLM check (fine-tune on competition-level vs grade-school math and cross-evaluate) is the natural follow-up and is not part of this experiment.
