# g2: Cross-domain. Does training only on Putnam-style maths buy poetry?

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/gpu/g2_cross_domain_math_vs_poetry/PLAN.md>

Status: pre-registered plan, written 10-18-2026, not run. Same freeze rule as g1: this file, `config.json` and `../common/` are committed unchanged before the pilot, and Sections 1–4 do not change after it. Shared machinery and dispatch: [../README.md](../README.md).

## 1. Question

Brando, 10-18-2026: *"if we overtrain on say putnam questions, it shouldn't really be able to do poetry…or something? even if putnam is considered 'harder'"*

This is the strong form of the alignment hypothesis: when the target sits outside the training support, no amount of "harder" data should buy it. Exp 13-small tested only the weak form (alignment varied within one concept family, every target reachable) and found that alignment does not dominate (elasticity ratio 0.80). Exp 17's disjoint-support arm (AT-437) is the synthetic version of this question; g2 is the real-model version.

## 2. Prediction (written before any run) and what would falsify it

Transfer ratio on held-out poetry, for arm X at the final checkpoint:

R_X = (bits/byte of base − bits/byte of X) / (bits/byte of base − bits/byte of P)

R = 0 means X bought nothing for poetry, R = 1 means as much as training on poetry itself, R < 0 means X made poetry worse.

- **Prediction.** Hard maths buys essentially no poetry (R_H ≤ 0.10, and very likely below 0: maths-only training costs some poetry modelling). "Harder" maths is no more general than easy maths (R_H − R_E ≤ 0.05). Web text (N) transfers more than either. Poetry training buys no maths (P's GSM8K and MATH-500 within about 1 point of base).
- **Falsified** if R_H's 95% interval lies above 0.25 (hard maths transfers a quarter of the in-domain gain or more), or if R_H − R_E's interval lies above 0.05 with every seed agreeing (difficulty buys generality).

## 3. Gates

All at step 610 (40.0M tokens). Interval = 95% bootstrap over the 1,000 held-out poetry documents (byte-weighted, 10,000 draws) of the seed-mean ratio.

| Gate | Quantity | SUPPORTED if | FALSIFIED if |
|---|---|---|---|
| B1 hard maths buys no poetry | R_H | ≤ 0.10 and interval upper bound < 0.25 | interval lower bound > 0.25 |
| B2 harder maths is no more general | R_H − R_E | interval upper bound ≤ 0.05 | interval lower bound > 0.05 and all seeds > 0.05 |

Descriptive (predicted directions, no verdicts): B3, R_H < 0 and R_H < R_N; B4, P − base on GSM8K and MATH-500 accuracy, and each arm − base on HellaSwag, with item intervals.

## 4. Design

**Model, optimiser, budget, seeds:** identical to g1 (Qwen2.5-0.5B main, 1.5B scale; full fine-tune; 610 steps × 65,536 tokens = 40.0M; constant learning rate from the shared pilot; seeds 0, 1, 2).

**Data.** g1's pools plus poetry:

| Pool | Source | Size | Licence |
|---|---|---|---|
| L1 grade-school maths | NuminaMath-CoT `gsm8k` + `orca_math` | ~160k | Apache-2.0 |
| L3 olympiad maths | NuminaMath-CoT `olympiads` + `aops_forum` + `amc_aime` | ~185k | Apache-2.0 |
| web | FineWeb-Edu sample-10BT, first shard, 150k documents | ~150k | ODC-By 1.0 |
| poetry | Gutenberg Poetry Corpus (`biglam/gutenberg-poetry-corpus`, 3.1M lines from 1,191 books), grouped into 32-line documents | ~95k documents | CC0-1.0 (public-domain texts) |

Held-out poetry comes from whole books never trained on (books are assigned to held-out by hash until 1,000 documents are reached), so no poem is split between training and test.

**Arms:**

| Arm | Mixture | Role |
|---|---|---|
| E | 100% L1 maths | easy maths (shared with g1) |
| H | 100% L3 maths | hard maths (shared with g1) |
| N | 100% web | general text (shared with g1) |
| P | 100% poetry | in-domain reference, defines R = 1 |

**Evaluation.** The primary metric is bits per byte on held-out poetry (tokenizer-independent; measures how well the model predicts unseen poems, not poem quality). Every run also gets GSM8K, MATH-500, HellaSwag and all held-out bits-per-byte sets (g1, Section 4). No generation-quality judge for poetry: an LLM judge would add its own noise and bias, and the question is transfer of modelling ability.

**Grid:** 4 arms × 3 seeds = 12 runs + 1 base = 13; only P is new beyond g1.

## 5. Tiers, cost and wall-clock (`runs.py cost`, A100-80GB, 35% utilisation)

| Tier | Runs | A100-hours | Wall-clock | Gates? |
|---|---|---|---|---|
| pilot (P at the default learning rate, 10M tokens, + base) | 2 | ~0.4 | ~30 min | no |
| seed-0 first look | 5 | ~2.9 | ~1 h | no |
| **main** (0.5B, 3 seeds) | 13 | **~8.4** (only ~2.2 beyond g1 when run together) | ~2 h on 8 GPUs | **yes** |
| scale (1.5B) | 13 | ~25 | ~4–5 h on 8 GPUs | replication |

## 6. Dispatch

[../README.md](../README.md), Section "Dispatch", with `PLANS=g2` alone or `PLANS=g1,g2,g3` (recommended: E, H and N are shared).

## 7. Analysis and verification

```bash
python experiments/gpu/g2_cross_domain_math_vs_poetry/analyze.py --out-root $OUT --tier main
python experiments/gpu/common/verify.py --plan g2 --tier main --out-root $OUT \
  --report-dir experiments/gpu/g2_cross_domain_math_vs_poetry/results_main
```

Outputs: `results_main/results.json`, `RESULTS_draft.md`, and `g2_cross_domain.png/.pdf` (held-out poetry bits per byte versus tokens for each arm with the base line, and a bar chart of R_E, R_H, R_N with intervals and the 0.25 falsification line). On planted fake runs the analysis recovered R_H = −0.061 (planted −0.06) and R_N = 0.099 (planted 0.10), and `verify.py` matched to 1e-9.

## 8. Threats to validity

- **Gutenberg poetry is old.** 19th-century verse is far from modern poetry; the held-out set is the same distribution as P's training set, which is what the ratio needs.
- **Bits per byte is modelling, not writing.** A model can predict poems better without writing better ones. That is the right measure for "transfer of what was learned", not for "can it write poetry".
- **Forgetting versus transfer.** Negative R mixes "learned nothing" with "forgot some"; B3 compares against N to separate content from generic drift.
- **Base exposure.** Qwen2.5 has seen Gutenberg-like text in pretraining, so P's gain (the denominator) may be modest; the ratio stays well defined as long as P improves on base, which the pilot checks.

**TLDR-end:** [tdc: GPU plan g2] Add a poetry-only arm to g1's maths and web runs and measure, on held-out Gutenberg poetry, how much of the in-domain gain hard maths transfers (R_H), with pre-registered gates for no transfer (R_H ≤ 0.10) and for difficulty buying no generality (R_H − R_E ≤ 0.05); ~2.2 A100-hours beyond g1 at 0.5B.
