# CKPT: GPU plans g1–g3 (hard-to-easy, cross-domain, LLM elasticity)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/gpu/CKPT.md>

Status: dated log of the GPU plans; nothing has run on a GPU. Write a line for every step of the order in [README.md](README.md): freeze commit, data manifest, pilot, learning-rate selection, disclosure commit, each tier, every failed or moved-aside run, analysis and `verify.py`. Times are US Pacific.

## 10-18-2026 (written on the CPU box, before any GPU run)

- Plans written: g1 (`g1_hard_to_easy_putnam_style_vs_gsm8k/PLAN.md`), g2 (`g2_cross_domain_math_vs_poetry/PLAN.md`), g3 (`g3_llm_complexity_vs_alignment_elasticity/PLAN.md`), with the shared code in `common/` and launchers in `launch/`.
- CPU smoke test (`common/smoke_test.sh` stages 1–2; toy data, `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, 8 steps): data prep, training and every evaluation (bits per byte, greedy generation with both graders, HellaSwag) ran for runs with one, two and three data pools; realised token shares matched the mixtures exactly (for example 0.1025 / 0.8975 for a 0.1 / 0.9 mixture at toy scale, within one document). Five smoke runs finished (grade-school only; grade-school + web; the three-level within-family mixture, two seeds of each mixture). Not yet run on CPU, because the machine was busy with other experiments: the base-model evaluation, the poetry pool, the web-only arm and the capped repeated-hard pool (Hrep). The GPU pilot exercises all four; run `common/smoke_test.sh` stage 2 first if a CPU is free.
- `train.py` refuses to reuse a DONE run that lacks generation evaluations a later plan needs (tested by hand on a smoke run).
- Planted-effect test (`common/smoke_test.sh` stage 3; fake runs with known answers): g1 recovered E − H on GSM8K 6.97 (planted 7), H − E on MATH-500 levels 4–5 7.17 (planted 6, on a 262-item subset), forgetting 3.89 (planted 3, biased upward by picking the best checkpoint, as PLAN.md says), H − Hrep 5.86 (planted 6); g2 recovered R_H −0.061, R_E −0.042, R_N 0.099 (planted −0.06, −0.04, 0.10); g3 recovered E 0.794, e_K 0.993, e_α 0.789 (planted 0.8, 1, 0.8), 8% of cells censored. `verify.py` printed VERIFY PASS for all three.

## Pilot first look, g1 only (pre-registered 10-18-2026 23:58, before any GPU run)

The pilot selects the learning rate and is not gated. Because it trains E and H anyway, this is the first-look reading of g1 it allows, fixed before it runs.

- Run: `TIER=pilot PLANS=g1` on Sherlock (7 runs: E and H at 5e-6, 1e-5, 2e-5, plus base; seed 0; ~10M tokens), at this commit.
- Read at the learning rate `runs.py select-lr` picks (its rule, unchanged), at the last pilot checkpoint (step 152).
- Prediction: (P1) E − H on GSM8K ≥ +3 points; (P2) H − E on MATH-500 levels 4–5 > 0; (P3) H's GSM8K is within 3 points of base or below it (hard-only data buys no grade-school maths at this scale).
- Readout: paired difference E − H over the 1,319 GSM8K items (and over MATH-500 levels 4–5) with a 95% paired bootstrap interval (10,000 resamples, seed 0). The effect is "visible" only when the interval excludes 0. When it includes 0 the pilot is too small to say, and is reported that way, never as a null. The sign at the other two learning rates is reported as a descriptive robustness check. One seed only: a visible pilot difference is a reason to run the seed-0 first look, not a result.
- Budget: capped at ~2 A100-hours expected; each run has a 40-minute wall limit.

## Freeze

- Freeze commit: `<sha>` (date, time). No GPU run existed before it.
- Data: `manifest.json` SHA-256 `<...>`; pool and evaluation counts; g3 complexity values K (median held-out solution tokens) L1 `<>`, L2 `<>`, L3 `<>`.

## Pilot disclosure

- Pilot runs: `<n>/9 DONE`, cluster and GPU type, measured training tokens per second per GPU `<>`, generation minutes per checkpoint `<>`.
- `runs.py select-lr` output (paste in full): `<...>`; selected learning rate `<>` (1.5B: half).
- Disclosure commit: `<sha>`. It changes only `common/suite.json` `selected_lr` and this file.

## Tiers

- `<date time>` seed-0 first look / main / scale: submitted (job ids or SNAP host and GPUs), finished, failures and what was done about each.

## Analysis

- `<date time>` analyze.py and verify.py for g1, g2, g3: verdicts, VERIFY PASS/FAIL.

**TLDR-end:** [tdc: GPU plans CKPT] Log for the GPU plans: CPU smoke and planted-effect tests passed on 10-18-2026; the freeze, pilot disclosure, tiers and verification are filled in as they happen.
