# g1 pilot (Sherlock, 10-19-2026)

**TLDR:** At pilot scale the hard-to-easy effect is not visible. At the selected learning rate, easy-only minus hard-only on GSM8K is −2.58 points (95% CI −5.38 to +0.30), the opposite sign to the prediction, and the interval includes 0, so the pilot is too small to say. P3 is refuted: hard-only data raised GSM8K by 15.1 points over the base model, about as much as easy-only did.

Qwen2.5-0.5B base, seed 0, 152 steps × 32 × 2048 tokens (about 10M tokens), one H100 80 GB per run. E = grade-school pool (L1), H = olympiad pool (L3). Readout at the last checkpoint, paired bootstrap over items (10,000 resamples, seed 0), as pre-registered in `../../CKPT.md` (freeze `4b94927f`).

| run | GSM8K | MATH-500 | MATH-500 L4–5 | L1 held-out bpb | L3 held-out bpb | HellaSwag |
|---|---|---|---|---|---|---|
| base | 29.34 | 18.4 | 5.73 | 0.1948 | 0.3659 | 48.40 |
| E, 5e-6 | 41.85 | 26.0 | 11.07 | 0.1344 | 0.3794 | 49.20 |
| E, 1e-5 | 42.61 | 26.8 | 13.74 | 0.1342 | 0.3881 | 49.45 |
| E, 2e-5 | 44.20 | 21.2 | 8.02 | 0.1364 | 0.3980 | 48.70 |
| H, 5e-6 | 44.43 | 25.6 | 12.98 | 0.2125 | 0.3468 | 48.95 |
| H, 1e-5 | 42.38 | 26.2 | 13.36 | 0.2172 | 0.3473 | 48.90 |
| H, 2e-5 | 39.50 | 20.2 | 10.31 | 0.2274 | 0.3507 | 47.70 |

Pre-registered readout at the selected rate, 5e-6 (`select_lr.txt`), point and 95% CI in points:

- P1, E − H on GSM8K ≥ +3: −2.58 (−5.38, +0.30). Not supported; the interval includes 0, so too small to say.
- P2, H − E on MATH-500 levels 4–5 > 0: +1.91 (−3.05, +6.87), 262 items. Right sign, interval includes 0, too small to say.
- P3, H within 3 points of base on GSM8K or below: H − base = +15.09 (+12.13, +17.97). Refuted.
- Robustness, E − H on GSM8K at the other rates: 1e-5 +0.23 (−2.50, +3.03); 2e-5 +4.70 (+1.67, +7.58). The sign changes with the learning rate.

Descriptive, not pre-registered: on likelihood the transfer is one-way. H made grade-school held-out text less likely than the base model (L1 bpb 0.2125 vs 0.1948) while E made it much more likely (0.1344), yet both raised GSM8K accuracy by 12–15 points. The accuracy gain from either pool may be mostly learning the answer format, which a base model at 29% is limited by; the bpb gap is where hard-only failing to teach easy material shows.

Cost: 7 runs, 3.58 H100-hours (28–33 min each, of which training was 230 s; GSM8K and MATH-500 generation took the rest). Failed attempts before it: about 0.07 GPU-hours (`attempts.txt`). Files: `pilot_report.json` (from `pilot_report.py`), `runs/runs/*/metrics.jsonl` (per-item correctness), `sherlock_inventory.txt`, `submit.log`.
