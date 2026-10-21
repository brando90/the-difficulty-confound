# CKPT: GPU plans g1–g3 (hard-to-easy, cross-domain, LLM elasticity)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/gpu/CKPT.md>

Status: dated log of the GPU plans; the g1 pilot ran on Sherlock on 10-19-2026. Write a line for every step of the order in [README.md](README.md): freeze commit, data manifest, pilot, learning-rate selection, disclosure commit, each tier, every failed or moved-aside run, analysis and `verify.py`. Times are US Pacific.

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

- 10-19-2026 07:45 result (read from `g1_hard_to_easy_putnam_style_vs_gsm8k/results_pilot/`): at the selected rate 5e-6, P1 E − H on GSM8K −2.58 points (95% CI −5.38, +0.30): not supported, too small to say. P2 H − E on MATH-500 levels 4–5 +1.91 (−3.05, +6.87): too small to say. P3 refuted: H − base on GSM8K +15.09 (+12.13, +17.97). E − H at 1e-5 +0.23 (−2.50, +3.03), at 2e-5 +4.70 (+1.67, +7.58): the sign depends on the learning rate. Cost 3.58 H100-hours, of which about 85% was generation for evaluation.

## g1 3-seed check, tier `seeds3` (pre-registered 10-20-2026, before any of its runs)

The pilot left two open readings: the GSM8K sign of E − H changes with the learning rate (−2.58 at 5e-6, +4.70 at 2e-5, one seed), and hard-only data bought about as much GSM8K as easy-only data, which may be the answer format rather than maths. This check repeats E and H at both rates with three seeds and adds a format-only arm. Nothing here changes the frozen pools, evaluation sets, prompt format or the earlier tiers; the tier, the arm and the grader are additions, and the run ids of every other tier are unchanged (`runs.py list --tier main --plans g1,g2,g3` gives the same output before and after this commit).

- Runs (16, `runs.py list --tier seeds3 --plans g1`): E and H at 5e-6 and 2e-5, seeds 0, 1, 2 (12 runs); F at 5e-6, seeds 0, 1, 2 (3 runs); base (1). Same data (`data/v1`, manifest SHA-256 `168634…d997`, the pilot's), model (Qwen2.5-0.5B), budget (152 steps × 32 × 2048 tokens, about 10M) and optimiser as the pilot. The seed sets the data order. Seed 0 of E and H has the pilot's run ids and is retrained, not reused, so that every arm is graded by the same grader; the pilot's seed-0 numbers are shown beside it, descriptive only (GPU training is not bit-reproducible).
- Format-only arm F (`g1_.../config.json`, `train.py` `answer_only`): the E and H pools' problems, 50/50 by tokens, each solution replaced by `The answer is $\boxed{<gold>}$.` (documents without a `\boxed{}` answer are dropped and counted in `summary.json`). It teaches the Problem/Solution layout, the boxed answer and stopping, with no worked reasoning, at the same 10M tokens and 152 steps (so it sees more, shorter problems than E or H). Learning rate 5e-6 only; the 2e-5 arms minus F are reported with that mismatch stated.
- Grader (`common/evals.py` `generate_math_fast`): the pilot's prompts, greedy decoding, 768-token cap, stop string and both graders (Math-Verify, then the last-boxed/last-number fallback for agreement), decoded in batches of up to 512 prompts in rounds of 256 tokens: a row that ends (EOS, stop string, cap) leaves, and the rest continue from their own token ids. On the pilot this part was 20–23 of each run's 28–33 minutes. Measured per run: generation seconds, and per item generated tokens, whether it stopped by itself and whether it gave a `\boxed{}` answer. Bits per byte only at step 152. Final weights saved (`--save-final`) so any run can be re-graded without retraining (`seeds3/regrade.py`).
- Grader check (gate, read before anything else): the new base-model evaluation against the pilot's base evaluation, item by item. Pass if, on GSM8K and on MATH-500 separately, the paired 95% interval of new − pilot includes 0 and |new − pilot| ≤ 1.5 points. On a fail, every run is re-graded from its saved weights with the pilot's engine (`regrade.py --engine pilot`, then `seeds3_report.py --regrade pilot`) and only that is read. (The base comparison also crosses torch 2.9.1 → 2.6.0, the venv Sherlock can install; the CPU check below isolates the decoding change.)
- CPU check of the decoding change (10-20-2026 19:00, after the pre-registration commit 8d62e5af, before any run; `g1_.../seeds3/grader_check/`): `generate_math_fast` against the pilot's `generate_math`, token for token, on 170 real prompts (GSM8K test every 11th item, MATH-500 every 10th) with a tiny random Qwen2 (`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, fp32, CPU), under four settings of stop strings, token cap, round length and batch-token limit, including rows that stop inside a round: 170/170 identical generations and identical grades in all four. The batched decoding is therefore the pilot's greedy decoding, not an approximation of it; the Sherlock base-model gate above still checks the GPU, torch version and long generations.
- Prediction, written before any run: (S1) at 5e-6, E − H on GSM8K stays near the pilot's value and does not reach +3 (P1 of the pilot is not supported again); (S2) at 2e-5, E − H is positive; (S3) the sign flip is therefore not "real" in the sense below, because at 5e-6 the interval will include 0; what is real is that a higher rate hurts H more than E on GSM8K (interaction > 0); (S4) F falls below the base model on GSM8K, because answer-only training teaches answering without working, so E − F and H − F are large and positive, and F does not by itself show how much of the gain is format.
- Readout, at step 152, in points. For each contrast: the three per-seed differences; their mean; the primary 95% interval, a paired bootstrap over items (10,000 resamples, seed 0) of the per-item difference averaged over the three seeds; and, secondary, the t interval over seeds (2 degrees of freedom). Contrasts: E − H on GSM8K and H − E on MATH-500 levels 4–5 (262 items), at each rate; each of E and H at each rate minus F, on GSM8K and on MATH-500 levels 4–5; each arm minus base, and F minus base, on GSM8K; the interaction (E − H at 2e-5) − (E − H at 5e-6) on GSM8K. Descriptive: stopped and boxed rates and mean generated tokens per run, held-out bits per byte on L1 and L3, HellaSwag.
- "The sign flip is real" only if all three hold: at 5e-6, E − H on GSM8K < 0 with its item interval excluding 0 and all three seeds negative; at 2e-5, E − H > 0 with its interval excluding 0 and all three seeds positive; the interaction is > 0 with its interval excluding 0. If only the interaction holds: "learning-rate dependent, sign not established". Otherwise: "not real at this scale".
- Other verdicts: P1 (E − H ≥ +3 on GSM8K at 5e-6, interval above 0, all seeds positive) supported or not; "H beats format-only on GSM8K at 5e-6" if H − F > 0 with its interval excluding 0 and all seeds positive; format share of H's gain = (F − base) / (H − base) at 5e-6, read as a format share only if F still works problems out (mean generated tokens on GSM8K ≥ 30 for all F seeds); otherwise F is reported as "answers without working" and the share is not interpreted.
- Code: `common/runs.py`, `common/train.py`, `common/evals.py`, `common/suite.json` (tier `seeds3`), `g1_.../config.json` (arm F), `g1_.../seeds3/` (env, sbatch, submit, report, re-grade). Sherlock: public `gpu` partition, `-C GPU_MEM:80GB` (H100 only; the job exits on any other GPU), 25-minute cap per run, own venv (python 3.12.1, torch 2.6.0+cu124 wheels, `--only-binary=:all:`), Slurm mail END,FAIL. Expected cost about 2.5 H100-hours (16 × ~9.5 min: training 230 s, generation about 3 min, the rest loading and bits per byte); worst case 16 × 25 min = 6.7 H100-hours.

## Freeze

- Freeze commit: `<sha>` (date, time). No GPU run existed before it.
- Data: `manifest.json` SHA-256 `<...>`; pool and evaluation counts; g3 complexity values K (median held-out solution tokens) L1 `<>`, L2 `<>`, L3 `<>`.

## Pilot disclosure

- Pilot runs: 7/7 g1 pilot runs DONE (E and H at 5e-6, 1e-5, 2e-5, plus base; seed 0), Sherlock public `gpu` partition, one H100_SXM5 80 GB per run (job array 48440728, node sh04-01n05, 10-19-2026 02:22–05:40). Measured training throughput 43,100–43,400 tokens per second per GPU (230 s of training per 10M-token run). Generation per checkpoint: GSM8K 10.3–14.0 min, MATH-500 9.4 min; bits per byte and HellaSwag under 1 min. Evaluation is about 85% of each run's 28–33 min wall time.
- Environment deviation (the repo tree was unchanged, run commit `d4fd7f93` has the freeze's `experiments/gpu` tree `15f99df5`): the `torch==2.14.1` pin has no wheel Sherlock (glibc 2.17) can install, so the runs used Sherlock modules python 3.14.2, torch 2.9.1 (CUDA 12.6), datasets 4.4.1, numpy 2.3.5, with transformers 4.57.6 (the repo pin) in a venv.
- Failed attempts, moved aside and not used (nothing deleted): (1) prep 48433204, no torch 2.14.1 wheel, 0 GPU time; (2) array 48435220, transformers 4.57.3 module raised OfflineModeIsEnabled loading the tokenizer, about 3.6 GPU-min; (3) array 48439388, CUDA out of memory on L40S 48 GB (fp32 0.5B, micro-batch 8 × 2048 logits), about 0.8 GPU-min. The run used `-C GPU_MEM:80GB`, the 80 GB class the plan was costed on.
- `runs.py select-lr` output (run on the downloaded pilot metrics, in full):

  ```
  lr 5e-06: E on L1 0.1344, H on L3 0.3468, mean 0.2406 bits/byte
  lr 1e-05: E on L1 0.1342, H on L3 0.3473, mean 0.2408 bits/byte
  lr 2e-05: E on L1 0.1364, H on L3 0.3507, mean 0.2436 bits/byte
  selected learning rate for qwen2.5-0.5b: 5e-06 (1.5B rule: 2.5e-06)
  ```

  Selected learning rate 5e-6 (1e-5 is within the 0.002 tie band, and ties go to the smaller rate); 1.5B: 2.5e-6.
- Disclosure commit: this commit. It changes only `common/suite.json` `selected_lr` and this file.

## Tiers

- `<date time>` seed-0 first look / main / scale: submitted (job ids or SNAP host and GPUs), finished, failures and what was done about each.

## Analysis

- `<date time>` analyze.py and verify.py for g1, g2, g3: verdicts, VERIFY PASS/FAIL.

**TLDR-end:** [tdc: GPU plans CKPT] Log for the GPU plans: CPU smoke and planted-effect tests passed on 10-18-2026; the freeze, pilot disclosure, tiers and verification are filled in as they happen.
