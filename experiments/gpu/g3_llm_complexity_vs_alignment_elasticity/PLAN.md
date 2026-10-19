# g3: LLM complexity versus alignment. The real-model counterpart of Exp 13's elasticity ratio

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/gpu/g3_llm_complexity_vs_alignment_elasticity/PLAN.md>

Status: pre-registered plan, written 10-18-2026, not run. Same freeze rule as g1: this file, `config.json` and `../common/` are committed unchanged before the pilot, and Sections 1–4 do not change after it. Shared machinery and dispatch: [../README.md](../README.md).

## 1. Question

The paper's claim is that the data a target needs depends more on alignment (how much of the training mix is the target) than on complexity (how hard the target is). Exp 13-small tested this on GINC-style hidden Markov model (HMM) mixtures and found the reverse: the elasticity ratio E = e_α/e_K, how far log-data moves per log-unit of alignment versus per log-unit of complexity, was 0.80 [0.79, 0.83], below the 1 the claim needs (`experiments/13_ginc_hmm_complexity_vs_alignment/RESULTS_ginc_hmm.md`).

g3 measures the same ratio on a real pretrained model, with maths difficulty as complexity and the share of target-level data in the mix as alignment. It answers whether 0.80 is a property of small synthetic HMMs or also of fine-tuning a language model.

## 2. Prediction (written before any run) and what would falsify it

For each target level k (complexity proxy K_k) and target share α, T*(k, α) is the number of training tokens needed to reach half of the improvement in held-out bits per byte that full alignment (α = 1) reaches at 40M tokens. Fit on all cells:

log T* = c + e_K · log K − e_α · log α, and E = e_α / e_K.

- **Prediction (the paper's alignment hypothesis, stated as for Exp 13):** E > 1, i.e. halving the target share costs more data than doubling target complexity. Exp 13-small found E = 0.80 in the synthetic setting, so a REVERSED verdict here would replicate the synthetic result on a real model, and that outcome is reported with equal prominence.
- **Falsified (REVERSED)** if E < 1 in the pooled fit and in every seed's own fit.
- **No test** if more than a third of cells are censored (T* outside the 0.33M–40M token range measured), or if complexity does not cost data at all (C1 fails), in which case e_K is not identified and E means nothing.

## 3. Gates

| Gate | Quantity | SUPPORTED if | FALSIFIED / REVERSED if |
|---|---|---|---|
| C1 complexity costs data (precondition) | M_K = T*(L3, α = 1) / T*(L1, α = 1), per seed | M_K ≥ 2 in every seed | M_K ≤ 1 in every seed |
| C2 alignment dominates | E = e_α / e_K, pooled fit and each seed's fit | E > 1 pooled and in every seed | E < 1 pooled and in every seed: **REVERSED** (complexity dominates, as in Exp 13-small) |

C2 is UNRESOLVED if the censored fraction exceeds 1/3, and is read only if C1 is SUPPORTED. Reported alongside, with no verdicts: E at q = 0.25 and 0.75 instead of 0.5; E with ordinal K (1, 2, 3) instead of token lengths; e_K and e_α separately; and, from the secondary arms, C3: E with within-family distractors (the rest of the mix is the other two maths levels) against E with web distractors (prediction: within-family E is lower, because other maths levels share structure with the target and so dilute it less).

## 4. Design

**Model, optimiser, budget, seeds:** identical to g1 (Qwen2.5-0.5B main, 1.5B scale; full fine-tune; 610 steps × 65,536 tokens = 40.0M; constant learning rate from the shared pilot; seeds 0, 1, 2).

**Complexity (K).** Three maths difficulty levels, each a held-out evaluation set of 1,000 problems with worked solutions. K_k is the median solution length in Qwen2.5 tokens over those 1,000 held-out solutions, computed by `prep_data.py` and written to the data manifest before any training (it is fixed by the data, not by any run). Ordinal K is the sensitivity check.

| Level | Source (NuminaMath-CoT, Apache-2.0) | Size |
|---|---|---|
| L1 grade-school | `gsm8k` + `orca_math` | ~160k |
| L2 competition | `math` + `synthetic_math` | ~175k |
| L3 olympiad | `olympiads` + `aops_forum` + `amc_aime` | ~185k |
| web (distractor) | FineWeb-Edu sample-10BT, first shard, 150k documents (ODC-By 1.0) | ~150k |

**Alignment (α).** The target level's share of training tokens, α ∈ {0.03, 0.1, 0.3, 1}; the rest is web text. Mixtures are exact by tokens.

**Arms.** Primary: 3 levels × 4 shares = 12 arms, named `L{k}_a{α}` (for example `L3_a0.1` is 10% olympiad, 90% web). `L1_a1` and `L3_a1` are the same runs as g1's E and H. Secondary (run with `SECONDARY=1`): 6 within-family arms `W_L{k}_a{0.1, 0.3}`, target share α with the remainder split equally over the other two levels.

**Measurement.** The only metric is bits per byte on the target level's held-out set, at steps 5, 10, 20, 40, 80, 160, 320, 610 (0.33M to 40M tokens). For level k, D(T) = bits per byte of the base model − bits per byte at T tokens, and D_max(k) = median over seeds of D at α = 1 and 40M tokens. T* is the first T with D(T) ≥ q · D_max(k), q = 0.5, interpolated linearly in log T between checkpoints; a cell is left-censored if the first checkpoint already passes and right-censored if the last does not. Censored cells are left out of the fit and counted. The pooled fit uses the per-cell mean of log T* over seeds (cells where any seed is censored are left out); each seed is also fitted on its own. No generation is needed except on the two runs shared with g1.

**Grid:** 12 arms × 3 seeds = 36 runs + 1 base = 37 (secondary: +18 = 55).

## 5. Tiers, cost and wall-clock (`runs.py cost`, A100-80GB, 35% utilisation)

| Tier | Runs | A100-hours | Wall-clock | Gates? |
|---|---|---|---|---|
| pilot (`L2_a0.1` at the default learning rate, 10M tokens, + base) | 2 | ~0.3 | ~30 min | no: checks the web mixture and that D(T) moves within range |
| seed-0 first look | 13 | ~5.4 | ~1.5 h on 8 GPUs | no |
| **main** (0.5B, 3 seeds) | 37 | **~15.8** (~22.7 with secondary) | ~3 h on 8 GPUs | **yes** |
| scale (1.5B) | 37 | ~45 (~65 with secondary) | ~8 h on 8 GPUs | replication |

Run alongside g1 and g2, the three plans together cost 24.0 A100-hours for main (49 runs), because six of g3's runs are g1's E and H.

## 6. Dispatch

[../README.md](../README.md), Section "Dispatch", with `PLANS=g3` (add `SECONDARY=1` for the within-family arms) or `PLANS=g1,g2,g3`.

## 7. Analysis and verification

```bash
python experiments/gpu/g3_llm_complexity_vs_alignment_elasticity/analyze.py --out-root $OUT --data-root $DATA --tier main [--secondary]
python experiments/gpu/common/verify.py --plan g3 --tier main --out-root $OUT --data-root $DATA \
  --report-dir experiments/gpu/g3_llm_complexity_vs_alignment_elasticity/results_main
```

Outputs: `results_main/results.json`, `RESULTS_draft.md` and `g3_elasticity.png/.pdf`: (1) D(T) curves per level and share, with the threshold; (2) log T* against log α per level, with the fitted lines; (3) E per seed and pooled against the line E = 1 and the synthetic 0.80. `verify.py` recomputes e_K, e_α and E from the raw metrics with its own code (normal equations by Cramer's rule) and must print `VERIFY PASS`. On planted fake runs (true e_K = 1, e_α = 0.8, E = 0.8), the analysis recovered E = 0.794, e_K = 0.993, e_α = 0.789 with 8% of cells censored.

## 8. Threats to validity

- **Three complexity levels.** e_K rests on three K values, so it is the weakest number here; the ordinal-K check shows how much rides on the length proxy, and C1 must pass before E is read.
- **K is a proxy.** Solution length mixes difficulty with verbosity of the NuminaMath rewrites. It is fixed before training and the ordinal check is reported next to it.
- **Pretraining exposure differs by level.** Qwen2.5 has seen more grade-school maths than olympiad maths, so D_max differs by level. The threshold is relative to each level's own D_max, which removes scale but not shape.
- **Censoring.** At α = 0.03 the hardest level gets only 1.2M target tokens; if those cells never reach half of D_max, they are right-censored and C2 may end UNRESOLVED. The seed-0 first look shows the censored fraction before the other seeds are spent.
- **Not the same estimand as Exp 13.** Exp 13 measured tokens to a fixed error target on from-scratch models; g3 measures tokens to half of the full-alignment gain on a pretrained model. A match in sign is meaningful; a match in value would be partly luck.

**TLDR-end:** [tdc: GPU plan g3] Fine-tune Qwen2.5-0.5B on 12 mixtures (three maths difficulty levels × target shares 0.03–1, rest web text), measure tokens to half the full-alignment gain in held-out bits per byte, and fit E = e_α/e_K with pre-registered gates (E > 1 alignment dominates; E < 1 in every seed reverses it, as Exp 13-small's 0.80 did); main ~15.8 A100-hours alone, 24.0 for all three plans.
