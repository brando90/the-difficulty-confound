## Plain-language summary

We trained three sizes of transformer (3M, 25M and 85M parameters) on made-up languages whose complexity we know exactly, mixed with ten other languages at four target shares, and measured how many training tokens each needed to get within 0.02 bits per word of perfect.

- **Complexity costs data at every size (G1).** The hardest concepts (128 hidden states) need 7.6 times the data of the easiest (8 states), in every seed.
- **Bigger models pay much less for complexity (G6, new).** That multiplier falls from 4.75 (S) to 2.66 (M) to 1.83 (L); L's is 0.39 of S's.
- **Alignment still does not dominate (G4).** Bigger models also care less about the target's share of the data, and that falls even faster than the complexity cost: for L, 20 times fewer target documents raise the data bill only 1.3 times, because L learns the target from the related languages. So by the fitted law, E = e_α/e_K falls with size (0.80 → 0.47 → 0.36). Read directly off the learning curves, E is about 1.0–1.2; the two readings disagree, so G4 is NOT_ESTABLISHED, not reversed.
- **Follow-up reanalysis (10-26-2026, [AT1346_censoring_reanalysis/](AT1346_censoring_reanalysis/README.md), exploratory, not gated):** keeping the runs that never reached ε* in a censored (Tobit) regression of the crossing times gives **E ≈ 1** (pooled 1.09 [1.06, 1.12] at ε* = 0.02, 1.03 at 0.05, 0.98 at 0.10), and the alignment cost rises with complexity (an α × n interaction). So the fitted law's 0.58 reflects the law fit; read off the crossing times, neither dial dominates.
- **Alignment acts early in a document, complexity late (G5)**, more strongly than in 13-small (0.93 vs 0.44).
- **The K law forecasts held-out rows 14% better in every seed (G2)**, but not on the held-out complexity level itself (0.581 vs 0.576), so G2 is NOT_ESTABLISHED. Complexity acts on both the coefficient and the exponent (κ 0.30, γ1 −0.019, both resolved), so C3 is unresolved (G3a, G3b).

## Predictions recorded before the gates were computed

Laolu's calls (CKPT, commit b1acee65, 10-24-2026 16:20, with 306 of 360 shards finished and no outcome read): G1 SUPPORTED ✓, G2 NOT_ESTABLISHED ✓, G3a NOT_ESTABLISHED ✓, G3b SUPPORTED ✗ (NOT_ESTABLISHED), G4 NOT_ESTABLISHED ✓, G5 SUPPORTED ✓, G6 SUPPORTED ✓: 6 of 7.

## Deviations and disclosures

1. **n = 128 concepts:** the frozen generator found none in 1,000 candidates; a per-level extension stream supplied them (PROTOCOL Section 7, 13-GPU implementation notes). Every other level and the band calibration are as frozen.
2. **Written after the freeze:** the 13-GPU code, gate interpretation (pooled verdict of record, every-seed rule), the N-law baseline and β starts. All committed before any GPU run (2c625111); the pilot ran at that commit rather than at the freeze, because the code did not exist at the freeze.
3. **Pilot:** the 2× multiplier, the top of the frozen grid, won for every size; a larger rate might do better.
4. **Hardware:** SNAP skampere1 (A100) and, from 10-24-2026 04:31, skampere2 (H200), all GPUs shared with other users; two GPU types (recorded per shard; every checkpoint replayed in float64).
5. **Operations:** `launch/snap.sh` exited silently (SIGPIPE under pipefail) and was replaced by `launch/snap_autopilot/`; the autopilot halted for 2 h 2 min after the pilot because `--check-only` rewrote a tracked receipt; no run, seed or checkpoint was dropped or rerun.
6. **Censoring:** at ε* = 0.02, half the runs (184 of 360) never reached ε* by their last checkpoint; model-free quantities cap them there and law-based ones extrapolate.
7. **ε_id = log2(1/α)/128 exactly:** the distractors give the targets almost no probability, so identification is trivial; the early-position alignment share (G5) is mostly ε_id plus slower early learning, not a test of identification difficulty.

## Which PROTOCOL Section 6 row matches

- "G1 SUPPORTED, G4 fails": complexity rivals or beats alignment in the fitted law at every size; η falls from 0.61 (S) to 0.10 (L), so with scale alignment works less through target-specific data and the related concepts carry most of the learning.
- G6 SUPPORTED: a bigger model shrinks the complexity multiplier, monotonically; the reverse outcome (≥ 1.5) is ruled out.
