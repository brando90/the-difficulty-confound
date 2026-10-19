# g1: Hard-to-easy. Does training only on Putnam-style maths buy grade-school maths?

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/PLAN.md>

Status: pre-registered plan, written 10-18-2026, not run. Freeze rule: commit this file, `config.json` and `../common/` unchanged as the freeze commit before the pilot; the pilot disclosure commit adds only the selected learning rate and measured throughput (`../common/suite.json` `selected_lr`, `../CKPT.md`). Sections 1–4 do not change after the freeze. Shared machinery and dispatch: [../README.md](../README.md).

## 1. Question

Brando, 10-18-2026: *"if we overtrain on putnam … you probably can't do gsm8k or something like that?"*

If difficulty were a ladder (a total order), training only on the hardest problems would also cover the easy ones: hard-only data would match easy-only data on GSM8K (Grade School Math 8K) and beat it on hard problems. The alternative is that the two ends need different skills (one-shot constructions versus reliable multi-step arithmetic bookkeeping), so each end is best served by its own data, and heavy hard-only training may even cost the easy end.

The synthetic counterpart (Exp 17's hard-to-easy arm on CPU, AT-437) asks the same question on HMM concepts. Exp 13-small already found that, within one concept family, alignment does not dominate complexity (elasticity ratio 0.80). g1 is the real-model test of the hard-to-easy question.

## 2. Prediction (written before any run) and what would falsify it

- **Prediction.** At 40M tokens, easy-only training (E) beats hard-only training (H) on GSM8K by at least 3 points, and H beats E on MATH-500 levels 4–5. That is a crossover: difficulty is not a ladder. H's GSM8K accuracy also falls between its best earlier checkpoint and 40M tokens, and repeating a small hard slice (Hrep) costs GSM8K more than fresh hard data does.
- **Falsified (difficulty is a ladder)** if the 95% interval of GSM8K(E) − GSM8K(H) lies entirely below 1 point while H still wins on the hard end.
- **No difficulty effect** if both A1 and A2 fail: the two pools are interchangeable at this budget.

## 3. Gates (all reported with equal prominence, whichever way they fall)

All at the final checkpoint (step 610, 40.0M tokens) unless stated. "Interval" = 95% paired bootstrap over test items (10,000 draws) of the seed-mean difference. "All seeds" = each of the 3 seeds on its own.

| Gate | Quantity | SUPPORTED if | FALSIFIED if |
|---|---|---|---|
| A1 easy end not subsumed | GSM8K accuracy E − H (points) | ≥ 3, interval lower bound > 0, all seeds > 0 | interval upper bound < 1 |
| A2 hard end needs hard data | MATH-500 levels 4–5 accuracy H − E | interval lower bound > 0, all seeds > 0 | interval upper bound < 0 |
| Crossover | A1 and A2 together | both SUPPORTED: **not a ladder** | A1 FALSIFIED and A2 SUPPORTED: **ladder** |
| A3a more hard training forgets | GSM8K of H at its best earlier generation checkpoint (2.6M or 10.5M tokens) minus at 40M | ≥ 2, lower bound > 0, all seeds > 0 | upper bound < 0 |
| A3b repeated hard slice forgets | GSM8K H − Hrep | ≥ 3, lower bound > 0, all seeds > 0 | upper bound < 0 |

Descriptive, no verdicts: M − E on GSM8K and M − H on MATH-500 levels 4–5 (does mixing cost anything?); E − N and H − N on both (content versus "any fine-tuning" and answer format); held-out bits per byte on each level's pool. A3a picks its comparison checkpoint on the seed-mean, which biases it upward; the analysis says so next to the number.

## 4. Design

**Model.** Qwen2.5-0.5B base (`Qwen/Qwen2.5-0.5B`, Apache-2.0, 494M parameters) in the main tier; Qwen2.5-1.5B base (Apache-2.0) in the scale tier. Full fine-tuning (no LoRA), so forgetting is measured on the whole network. SmolLM2-1.7B (Apache-2.0, far less maths in pretraining) is the named robustness model if Qwen's GSM8K exposure turns out to dominate (Section 8).

**Training data** (all built by `../common/prep_data.py`, decontaminated against every test item by shared 13-word sequences):

| Pool | Source | Size | Licence |
|---|---|---|---|
| L1 grade-school (easy) | NuminaMath-CoT sources `gsm8k` + `orca_math` | ~160k problems with worked solutions | Apache-2.0 (GSM8K MIT, Orca-Math MIT) |
| L3 olympiad (hard, "Putnam-style") | NuminaMath-CoT sources `olympiads` + `aops_forum` + `amc_aime` | ~185k | Apache-2.0 |
| web (control) | FineWeb-Edu, `sample/10BT/000_00000.parquet`, first 150k documents | ~150k | ODC-By 1.0 |

Real Putnam problems number about 1,500 in total, too few to train on, so the olympiad/AoPS/AMC-AIME pool is the licensable stand-in. 1,000 problems per pool are held out (hash split, seed 20261018) for bits-per-byte evaluation.

**Arms** (each mixture is exact by tokens, to within one document):

| Arm | Mixture | Role |
|---|---|---|
| E | 100% L1 | easy-only |
| H | 100% L3 | hard-only |
| M | 50% L1 + 50% L3 | mixed control |
| N | 100% web | off-topic control: separates content from fine-tuning and answer format |
| Hrep | 100% L3, restricted to a fixed 2.5M-token slice, repeated ~16 times | Brando's literal "overtrain" |

Matched compute: every arm trains for exactly the same number of steps and tokens with the same optimiser.

**Budget and optimiser.** 610 steps × 32 sequences × 2,048 tokens = 40.0M tokens per run. Documents formatted `Problem: …\nSolution: …`, separated by EOS and packed. AdamW (β = 0.9, 0.95; no weight decay; gradient clip 1.0), 10 warmup steps then a **constant** learning rate, so every checkpoint is a comparable stopping point. bf16 autocast with fp32 master weights, gradient checkpointing. Learning rate: chosen by the pilot from {5e-6, 1e-5, 2e-5} by the pre-registered rule in `runs.py select-lr` (lowest mean held-out bits per byte of E on L1 and H on L3 at 10M tokens; ties within 0.002 go to the smaller rate); the 1.5B tier uses half of it.

**Evaluation.** Bits per byte on all held-out sets at steps 5, 10, 20, 40, 80, 160, 320, 610 (0.33M–40M tokens). At steps 40, 160 and 610 (2.6M, 10.5M, 40M tokens), greedy decoding (up to 768 new tokens, stop at `\nProblem:`) on GSM8K test (1,319, MIT) and MATH-500 (500, MIT; 262 at levels 4–5), graded with Math-Verify against the gold answer, plus an independent last-`\boxed{}`/last-number grader whose agreement is reported; HellaSwag (first 2,000 validation items, MIT) by length-normalised likelihood. The untrained base model is evaluated once with the same prompts.

**Seeds and grid.** Seeds 0, 1, 2 (data order and shuffles; the pretrained weights are fixed). Grid: 5 arms × 3 seeds = 15 runs, plus 1 base evaluation = 16 runs. E, H and N are shared with g2, and E and H are g3's (L1, α = 1) and (L3, α = 1) cells, so they train once for all plans.

## 5. Tiers, cost and wall-clock

Cost is `runs.py cost`: A100-80GB hours at 35% model FLOPs utilisation for training, plus about 0.1 h per generation checkpoint at 0.5B. The pilot measures real throughput; rescale by the measured tokens per second.

| Tier | Runs | A100-hours | Wall-clock | Gates? |
|---|---|---|---|---|
| pilot (0.5B, seed 0, 10M tokens, E and H at 3 learning rates) | 7 | ~1.6 | ~1 h on 4–6 GPUs | no: selects the learning rate, checks every evaluation |
| seed-0 first look (main, `SEEDS=0`) | 6 | ~3.6 | ~1 h on 6 GPUs | no: a preview; its runs are reused by main |
| **main** (0.5B, 3 seeds, 40M tokens) | 16 | **~10.4** | ~2 h on 8 GPUs | **yes** |
| scale (1.5B, 3 seeds) | 16 | ~31 | ~5 h on 8 GPUs | yes, as a replication at 3× parameters |

With single-digit Sherlock hours, run the pilot and the seed-0 look on Sherlock and the main tier on SNAP (skampere1/2/3) or on the NAIRR allocation once it lands.

## 6. Dispatch

Exact commands are in [../README.md](../README.md), Section "Dispatch". For this plan alone use `PLANS=g1`; to share runs with g2 and g3 (cheaper overall) use `PLANS=g1,g2,g3`.

## 7. Analysis and verification

```bash
python experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/analyze.py --out-root $OUT --tier main
python experiments/gpu/common/verify.py --plan g1 --tier main --out-root $OUT \
  --report-dir experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/results_main
```

`analyze.py` writes `results_main/results.json`, `RESULTS_draft.md` (gate table and arm table) and `g1_hard_to_easy.png/.pdf`: GSM8K accuracy versus tokens per arm, MATH-500 levels 4–5 versus tokens, and held-out bits per byte on L1 and L3. `verify.py` checks every run's DONE hash and completeness and recomputes A1, A2 and A3b in plain Python; it must print `VERIFY PASS` before anything is reported. Both were tested on planted fake runs (`../common/fake_runs.py`): planted E − H = 7 points was recovered as 6.97, H − Hrep = 6 as 5.86.

## 8. Threats to validity

- **Qwen2.5 has seen a lot of GSM8K-style data.** The base model already solves part of GSM8K, so g1 measures movement from a strong prior, and forgetting is easier to see than learning. The N arm and the base evaluation bound this; SmolLM2-1.7B is the fallback model if E barely moves above base.
- **"Putnam-style" is a proxy.** The hard pool is olympiad/AoPS/AMC-AIME, with solutions rewritten by the NuminaMath team; many are proofs without a single final answer. That affects only training content, not grading, which is on GSM8K and MATH-500.
- **Answer format.** Fine-tuning teaches `\boxed{}` answers, which helps grading for every trained arm relative to base; comparisons between trained arms are unaffected, and N shows the format-only gain.
- **One learning rate for all arms.** Chosen on E and H together, so neither is favoured; a per-arm optimum could differ.
- **Three seeds.** Intervals cover test-set sampling; training noise is covered only by the all-seeds rule.

**TLDR-end:** [tdc: GPU plan g1] Fine-tune Qwen2.5-0.5B (then 1.5B) on easy-only, Putnam-style hard-only, mixed, web-control and repeated-hard data at matched 40M tokens, and test with pre-registered gates whether hard-only training subsumes GSM8K (ladder) or each end needs its own data (crossover); pilot ~1.6 A100-hours, main ~10.4.
