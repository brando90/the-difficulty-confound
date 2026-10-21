# Coding-agent prompt: run Experiment 16 end to end

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/16_carry_alignment_crossed_arithmetic/PROMPT.md>

Paste the block below into Claude Code or Codex at the repository root. It is self-contained and covers both arms. The coding agent runs the CPU arm (16a) on a 4-core box. A person (Laolu or Brando) launches the GPU arm (16b), the primary arm, on SNAP or Sherlock with the scripts in `launch/`. The agent builds and tests everything, runs 16a, and then either merges the 16b results or stops at the documented point.

```text
ROLE AND GOAL
You are a careful research engineer. Run Experiment 16 of the repository
https://github.com/brando90/the-difficulty-confound end to end.

What the experiment does: cross carry-chain length c with the target stratum's
share alpha of the training data, in addition problems.
- c is the longest run of consecutive carries in a problem. It is a structural
  demand, NOT Kolmogorov complexity. The law variable is s = c + 1.
- Fit the program's law family and test, with numeric gates:
  - whether chain length costs data (C1);
  - whether alignment dominates it (C2);
  - whether it moves the coefficient or the exponent, compared with Experiments
    11/12 (C3).

There are two arms:
- 16b (primary, GPU): fresh 8-digit problems, three transformer sizes, plus a
  reversed-output control.
- 16a (secondary, CPU): the 3-digit multilayer perceptron (MLP) setup of
  Experiments 06/08.

Your job: build, test, pilot, freeze, run 16a, verify, and report. Then either
merge the 16b results (if they exist) or stop at the documented point. In every
case open a pull request and do not merge it; merging waits on Brando.

BRANCH
git checkout main && git pull && git checkout -b exp16-carry-crossed

READ FIRST, IN THIS ORDER
1. experiments/16_carry_alignment_crossed_arithmetic/PROTOCOL.md: the design and
   the contract. Sections 2-5 must not change after your freeze commit.
2. experiments/ideas/complexity_vs_alignment/README.md: the program, the
   notation, the law names (M0_scale, M_align_eff, M_align_add, M_K_coef,
   M_K_exp, M_K_both), claims C1-C3, and the shared rules.
3. experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md and
   experiments/ideas/predictability_bridge/README.md ("Where Kolmogorov complexity
   belongs"). Never call carry-chain length complexity or K in prose; it only
   fills the K column of the shared output contract.
4. experiments/archive/06_compositional_carry_marginal_matched/ (PROTOCOL.md,
   RESULTS_compositional.md, expt_v1/addition.py: carry_chain, column_addition,
   model) and experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/
   (run.py: carry_cells, dataset; laws.py, independent_fit.py, analyze.py,
   verify.py, test_laws.py). 16a reuses this code: import it read-only, check the
   SHA-256 of each imported file in your manifest, and copy nothing.
5. experiments/16_carry_alignment_crossed_arithmetic/launch/ (README.md, snap.sh,
   sherlock.sbatch): your run.py must provide the interface these scripts call
   (see run.py below).
6. If it exists (main or archive): the frozen PROTOCOL.md of Experiment 11 (and
   12), Section 5. If their G1-G4 differ from this protocol's, align them BEFORE
   your freeze and say so in the freeze commit message and in CKPT.
7. Background: Lee et al. 2023, "Teaching Arithmetic to Small Transformers",
   https://arxiv.org/abs/2307.03381 (output formats; reversed output).

ENVIRONMENT (CPU box: 4 cores, 15 GB RAM, no GPU, no torch preinstalled)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
- Pin every installed version exactly in expt_v1/requirements.txt.
- In code, call torch.set_num_threads(1). Run at most 4 worker processes. Keep RAM
  under 12 GB.
- Record the torch version in config.json as "gpu_torch_version".
- Write expt_v1/requirements-gpu.txt: the same pins minus torch.

FILES TO CREATE under experiments/16_carry_alignment_crossed_arithmetic/expt_v1/

config.json: every constant in the PROTOCOL, with blocks "mlp" (16a) and
"transformer" (16b).

- Shared:
  - split_seed / task_sampling_seed 20261016; analysis_seed 29101626;
    bootstrap_draws 20000; n_starts 64.
  - Bounds: gamma (0,3]; gamma1 [-1,1]; eta [0,3]; kappa [-3,3]; beta (0,3].
  - rmse_reduction_threshold 0.10; bonferroni_alpha 0.025; g3_ratio 0.90;
    g3_gamma1_band 0.02; g6_ratio 0.67; g7_log_ratio 1.5;
    fallback_fail_share 0.25.
  - independent_fit_tol 1e-4; verify_tol_bits 1e-9.
- mlp:
  - Data: digits 3; minimum_pairs_per_answer_cell 28; test/calibration/probe per
    answer cell 4/2/2; train_per_answer 20.
  - Strata: nocarry=[cell 0] (c 0), isolated=[cells 1,2] (c 1),
    cascade=[cell 3] (c 2).
  - alpha_grid [0.1,0.33,0.6,0.9]; heldout_alpha 0.33; heldout_c 1.
  - widths [32,64,96,128]; heldout_width 96;
    expected_params [4328,10664,19048,29480].
  - Optimizer: lr 1e-3; weight_decay 1e-4; batch 256 with replacement;
    warmup 50; constant schedule.
  - checkpoints_updates [100,300,1000,3000].
  - seeds [16101,16202,16303,16404,16505,16606,16707,16808]; pilot_seed 16909.
  - epsilon_star 0.25.
  - workers 4; threads 1; max_wall_seconds 5400.
- transformer:
  - digits 8; vocab ["0".."9","+","=","$"]; seq_len 29; block 32.
  - primary_format "plain"; control_format "reversed"; control_size "6M".
  - train_levels [0,1,2,4,7]; probe_levels [3,5,6,8]; heldout_c 4;
    alpha_grid [0.05,0.15,0.4,1.0]; heldout_alpha 0.15; test_per_stratum 2000.
  - sizes 1M (5,128,4), 6M (8,256,8), 25M (8,512,8);
    expected_params [997376,6330112,25243136];
    expected_nonembedding [991616,6318592,25220096];
    lr [2e-3,1e-3,5e-4].
  - Optimizer: betas [0.9,0.95]; weight_decay 0.1; clip 1.0; batch 256;
    warmup 200; constant schedule; bf16 autocast for training.
  - T_checkpoints [2^16,2^18,2^20,2^22,2^24] examples.
  - seeds [16101,16202,16303]; pilot_seed 16919; pilot_c 3; pilot_alpha 0.4;
    pilot_lr_multipliers [0.5,1,2]; pilot_T 2^22;
    format_swap_threshold_bits 2.0.
  - epsilon_star 0.1; n_shards 240; data_workers 8.
  - lr_selected: null and primary_format_final: null (set only by the GPU pilot
    disclosure commit).

carry.py: shared exact arithmetic.
- column_addition and carry_chain, imported read-only from Experiment 06's
  addition.py (hash-checked).
- The chain-length dynamic program over (column, carry, current run, max run):
  - stratum counts for any digit width;
  - exact uniform within-stratum sampling, least-significant column first, with
    probabilities proportional to completion counts. Python integers, no floats,
    in the counts.
- encode(a, b, fmt) -> 29 token ids for fmt "plain"
  ($ a7..a0 + b7..b0 = s8..s0 $) or "reversed" ($ a7..a0 + b7..b0 = s0..s8 $),
  and answer_positions(fmt).
- Covariates per stratum: input-digit total-variation distance from the natural
  distribution, answer-digit entropy, number of carries, and exposure to
  chains >= k.

mlp_arm.py (16a):
- Imports Experiment 08's dataset() and carry_cells (hash-checked) and calls
  them with split seed 20261016.
- training_ids(seed, target, alpha):
  - exact cell totals (target 3600*alpha; each distractor stratum
    1800*(1-alpha); isolated split equally over cells 1 and 2);
  - exactly 20 per answer, by floor quotas plus a deterministic max-flow
    assignment of the remainders;
  - nested across alpha via one seeded permutation per (seed, answer, cell)
    pool.
- Logs the column-carry rates of every mix.
- The model is Experiment 06's MLP (60 -> w -> w -> 40).

transformer_arm.py (16b):
- nanoGPT-style decoder (pre-LayerNorm, learned positions, GELU, MLP 4d, tied
  embeddings, no dropout).
- Training stream: example j of stratum c in seed s comes from numpy PCG64
  seeded by (s, c, j). Exact Bresenham target schedule: example k is a target iff
  floor((k+1)*alpha) - floor(k*alpha) == 1. Non-target slots cycle through the
  other 4 training levels in a seeded permutation.
- Rejection of test pairs and their commuted twins, with a rejection counter.
- Loss on the 9 answer positions only.
- At start, asserts CUDA is available with compute capability >= 8.0 (bfloat16),
  and records the GPU name, driver and torch/CUDA versions.

run.py: command-line interface.
  --arm mlp|transformer
  --check-only
  --pilot
  --output DIR (refuses to overwrite an existing non-empty DIR, except to resume
    shards)
  --shard I (transformer: one shard)
  --pilot --pilot-job J (transformer: J = 3*i_size + i_lr, 0..8)
  --pilot-select DIR (transformer: apply the frozen pilot rules, print and write
    the chosen multipliers and the final primary format)
  --merge DIR (transformer: collect shards into results/transformer without
    checkpoints)
  --workers 4
- Shard map exactly as PROTOCOL Section 2.6:
  - primary: shard = 60*i_N + 12*i_c + 3*i_alpha + i_seed (0-179);
  - reversed control at 6M: 180 + 12*i_c + 3*i_alpha + i_seed (180-239).
  - If the pilot swapped formats, the "primary" shards use the reversed format
    and the "control" shards the plain format (read from
    transformer.primary_format_final). A shard refuses to start if lr_selected is
    unset.
- Each shard writes OUT/shards/<%05d>/ and, last, a DONE file with the SHA-256 of
  every file it wrote. Pilot job J writes OUT/pilot/<%05d>/ and DONE.
- --check-only (both arms) does all of the following and trains nothing:
  - builds the datasets;
  - asserts every count in the PROTOCOL (2,880 16a test pairs; 3,600 training
    pairs with 20 per answer and exact cell totals for every (seed, target,
    alpha); 2,000 16b test pairs per stratum);
  - checks disjointness (16b: including commuted twins);
  - asserts the DP total equals 10**16 and the 3-digit DP equals brute force;
  - asserts parameter counts;
  - prints the run counts, the shard map and a time estimate.
- --arm mlp --pilot: PROTOCOL 2.1 pilot (seed 16909, 6 runs) into
  results_pilot/mlp. Prints timing and the epsilon_star rule outcome.
- Evaluation at each checkpoint, in float64:
  - teacher-forced bits per answer and greedy exact match on every stratum's
    test pairs (16b: all 9 strata; 16a: all cells);
  - per-digit bits.
- Writes metrics.csv with the shared columns (seed, target_id, K, K_bits, alpha,
  T, step, excess_bits, accuracy, n_test) plus arm, format, N, c, s,
  eval_stratum, digit_bits_*. K holds s; K_bits is empty.
- Also writes tasks.json, data/, mixes.json, checkpoints/...npz (float32, no
  pickle) and manifest.json.
- Stops cleanly at max_wall_seconds (16a) with a resumable state.

laws.py:
- The six laws of PROTOCOL Section 4.1, with the N term.
- Fitting: least_squares "trf" on log2 epsilon, 64-start grid, bounds from config.
- Invalid-forecast rule.
- T_star at the largest model, M_s, M_alpha, the elasticities and epsilon_flip,
  as in PROTOCOL Section 4.3.

independent_fit.py: L-BFGS-B refit from a different start grid. Must agree to
1e-4 relative on every held-out RMSE; prints AGREE or the first disagreement.

analyze.py:
- Per-seed fits on fit rows. forecasts.csv is written and hashed before any
  held-out row is read.
- Held-out RMSE and the seed-block bootstrap.
- Gates G1-G7 and 16a-G1..G4 exactly as PROTOCOL Section 5, with the
  every-seed rule for 16b.
- G5: reads Experiment 11/12 gates.json from main or archive. If either is
  absent or undecided, the verdict is NOT_ESTABLISHED with the named reason.
- Sensitivity fits and mechanism analyses (Section 4.5).
- Writes fits.json, gates.json and bootstrap.npz.

verify.py: PROTOCOL Section 8. Prints PASS or the first failure.
- --arm mlp|transformer.
- --shard I --output DIR: verifies one transformer shard and writes VERIFIED next
  to DONE (the launch scripts use this in MODE=verify).

report.py: RESULTS_carry_crossed.md and figures.
- Figures:
  - epsilon vs T per c and alpha (log-log);
  - M_s and M_alpha vs epsilon;
  - per-digit excess, plain vs reversed;
  - probe strata vs c.
- One gate table: estimate, 95% interval, threshold, verdict.

test_carry.py, test_laws.py, test_arms.py: the unit tests below.

requirements.txt, requirements-gpu.txt: exact pins.

UNIT TESTS THAT MUST PASS BEFORE ANY RUN
1. Arithmetic:
   - column_addition equals Python addition on 10^5 random 8-digit pairs and on
     all 3-digit pairs;
   - carry_chain on [123,109,199,999] + [234,1,1,1] gives [0,1,2,3];
   - the 8-digit carry_chain matches a brute-force column walk on 10^5 pairs.
2. Dynamic program:
   - stratum counts for widths 1-4 equal brute-force enumeration;
   - 8-digit counts sum to 10**16, and the shares match the PROTOCOL table to
     4 decimals.
3. Sampler:
   - for width 3, every drawn pair has the requested c;
   - the empirical distribution over 10^6 draws per stratum passes a chi-square
     test against uniform-within-stratum (p > 0.001).
4. Formats:
   - encode/decode round trip for both formats;
   - the answer positions hold exactly the 9 sum digits in the right order;
   - for every 3-digit pair, the reversed-format carry into column i equals
     [a_{i-1} + b_{i-1} - s_{i-1} >= 9] (documents why reversed is the control).
5. Alignment:
   - 16a: exact cell totals and 20 per answer for every (seed, target, alpha);
     sets nested across alpha; disjoint from test/calibration/probe;
   - 16b: exact floor(m*alpha) targets after every prefix; non-target strata
     counts differ by at most 1; common random numbers across alpha.
6. Disjointness: no 16b training pair (or its commuted twin) is a test pair, on
   a reduced instance with forced collisions.
7. Models: parameter counts equal the config for all MLP widths and all
   transformer sizes; the npz save/load round trip is exact.
8. Laws:
   - every fitter recovers known parameters from synthetic data (noise sd 0.02;
     exponents within 5%);
   - independent_fit agrees to 1e-4;
   - invalid forecasts are detected.
If a test fails twice after honest fixes, write the blocker in CKPT_carry_crossed.md
and stop.

ORDER OF WORK
1. Build the files. Run the unit tests (python -m pytest -q expt_v1).
   Run run.py --arm mlp --check-only and run.py --arm transformer --check-only
   (CPU is fine for the transformer check: it trains nothing). Save both outputs
   in CKPT.
2. 16a pilot: run.py --arm mlp --pilot (seed 16909). Apply the epsilon_star rule
   of PROTOCOL Section 4.4. Write a "Pilot disclosure" paragraph at the end of
   PROTOCOL Section 2. Seed 16909 is never used again.
3. If Experiment 11's frozen G1-G4 differ from these, align them now.
4. FREEZE: commit PROTOCOL.md, config.json and the code with the message
   "Freeze Experiment 16 protocol". No gated run may exist before it. Push the
   branch. Record the freeze SHA in CKPT.
5. 16a full run under the wall cap:
     python expt_v1/run.py --arm mlp --output expt_v1/results/mlp --workers 4
   (384 runs).
6. verify.py --arm mlp must print PASS. If it fails, fix the cause, log it in
   CKPT, and rerun only what the fix touches; every rerun is logged.
7. 16b: you do not launch it. Write in CKPT the launch instructions from
   launch/README.md with the freeze SHA filled in.
   a. The person launching runs MODE=pilot (9 jobs) at the freeze SHA, then
      run.py --arm transformer --pilot-select <OUT>/pilot.
   b. ONE commit titled "Experiment 16 GPU pilot disclosure" then records
      transformer.lr_selected and transformer.primary_format_final in
      config.json, plus a "GPU pilot disclosure" paragraph in PROTOCOL
      Section 7. Sections 2-5 are untouched.
   c. The shard array (0-239) runs at that commit.
   - If results/transformer exists (merged with run.py --arm transformer --merge
     <OUT>): run verify.py --arm transformer, analyze and report it.
   - If not: RESULTS reports the 16a gates and states "16b: pending, not run";
     CKPT says the same; skip step 11 (archive). Open the pull request as
     "Experiment 16 (16a complete, 16b pending)".
8. analyze.py, then report.py (each arm that exists).
9. RESULTS_carry_crossed.md (in the experiment folder) must contain:
   - every gate (G1-G7, 16a-G1..G4) at equal prominence, SUPPORTED or
     NOT_ESTABLISHED, each with its number and 95% interval;
   - the fitted laws per seed;
   - the multiplier curves;
   - per-digit excess (plain vs reversed);
   - the probe strata;
   - the covariates (16a column-carry rates; 16b input-digit distance);
   - the sensitivity fits;
   - the PROTOCOL Section 6 row that matches;
   - every pilot decision, including whether formats swapped.
   It must say plainly that c is a structural demand, not K.
10. CKPT_carry_crossed.md: a dated log (MM-DD-YYYY) of every step, pilot, failure,
    rerun, decision and timing. ARTIFACT_SHA256.json over results/ and the
    reports. 16b checkpoints stay on cluster storage: record their SHA-256s (from
    DONE files) and do not commit them.
11. Archive (only when both arms are complete):
    - git mv experiments/16_carry_alignment_crossed_arithmetic
      experiments/archive/16_carry_alignment_crossed_arithmetic;
    - fix relative links;
    - add one row to experiments/README.md (setup and status, one line, numbers
      from RESULTS).
12. Push the branch and open a pull request with gh pr create.
    - Summary: at most 10 bullets, every gate verdict with its number.
    - Test plan: 2-3 bullets (unit tests, verify.py PASS, check-only).
    - Merging waits on Brando.

HARD RULES
- Never change a gate, threshold, estimator, seed, grid cell or count after the
  freeze commit. The only post-freeze edit allowed is the GPU pilot disclosure
  (config lr_selected and primary_format_final plus a PROTOCOL Section 7
  paragraph), by the frozen rules, in its own commit before any GPU shard.
- Never drop a seed, stratum, run, checkpoint or forecast. Failed or diverged runs
  stay in, flagged, and count as invalid forecasts where they enter a law.
- No hidden reruns: every run, including failed ones, is in CKPT.
- If a unit test fails twice, stop and write the blocker in CKPT.
- Bounded resources on the CPU box: 4 workers x 1 thread, RAM < 12 GB,
  max_wall_seconds 5400, artifacts < 2 GB.
- Never run 16b on the CPU box. Never open SSH connections in loops.
- Never modify Experiment 06/08/11/12 files; import read-only and hash-check.
- No calls to any model-provider API. No credentials in the repository.
- Cite the alignment-scaling paper only as "the alignment-scaling paper with
  Varun", linking https://github.com/madanva/alignment-scaling-laws.
- No tool or AI attribution in commits, docs or the pull request.
- Plain English; expand acronyms on first use; dates MM-DD-YYYY.

DELIVERABLE CHECKLIST
[ ] expt_v1/ code, config.json, requirements.txt, requirements-gpu.txt, tests,
    all passing
[ ] run.py --check-only output (both arms) saved in CKPT
[ ] 16a pilot disclosed in PROTOCOL.md; freeze commit before any gated run;
    freeze SHA in CKPT
[ ] results/mlp complete: tasks.json, data/<task_id>_{train,test}.npz, mixes.json,
    metrics.csv, checkpoints/, fits.json, forecasts.csv, gates.json,
    bootstrap.npz, manifest.json
[ ] verify.py --arm mlp prints PASS; independent_fit.py prints AGREE
[ ] results/transformer complete and verify.py --arm transformer PASS, OR
    "16b: pending, not run" stated in RESULTS and CKPT with launch instructions
[ ] RESULTS_carry_crossed.md with every gate at equal prominence and the figures
[ ] CKPT_carry_crossed.md (dated log)
[ ] ARTIFACT_SHA256.json
[ ] if both arms are complete: folder moved to
    experiments/archive/16_carry_alignment_crossed_arithmetic/, links fixed, row
    added to experiments/README.md
[ ] branch exp16-carry-crossed pushed, pull request open, not merged
```

**TLDR-end:** [tdc: Expt 16 prompt] A self-contained coding-agent prompt that builds the exact chain-length sampler, both addition arms and the law fitters, then pilots, freezes, runs and verifies the CPU MLP arm, reports every gate at equal prominence, and either merges the sharded GPU transformer arm or stops with it marked pending.
