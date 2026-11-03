# Coding-agent prompt: run Experiment 13 end to end

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/13_ginc_hmm_complexity_vs_alignment/PROMPT.md>

Paste the block below into Claude Code or Codex at the repository root. It is self-contained. It covers both arms. The CPU arm (13-small) runs on a 4-core box. The GPU arm (13-GPU) is launched by a person (Laolu or Brando) on SNAP or Sherlock with the scripts in `launch/`. The agent builds and tests everything and runs 13-small, then either merges the GPU results or stops at the documented point.

```text
ROLE AND GOAL
You are a careful research engineer. Run Experiment 13 of the repository
https://github.com/brando90/the-difficulty-confound end to end.

What the experiment does: train small decoder transformers on GINC-style mixtures
of hidden Markov model (HMM) "concepts". GINC is the Generative In-Context learning
dataset. The design crosses
  - target concept complexity (minimal hidden-state count n, certified exactly by
    process rank), with
  - the target's share alpha of the training tokens.
It scores excess loss in bits per token against two exact references, both
computed with the forward algorithm: the true concept (oracle) and the Bayes
mixture under the training prior.

The gates ask:
  - whether complexity costs data (C1);
  - whether alignment dominates complexity (C2);
  - whether complexity acts as a coefficient or as an exponent (C3);
  - whether alignment acts early in a document and complexity late.

There are two arms:
  - 13-small: CPU, one model size. You run it.
  - 13-GPU: three model sizes. A person launches it from launch/ on GPU nodes.

Your job: build, test, pilot, freeze, run 13-small, verify, and report. Then
either merge the GPU results (if they exist) or stop at the documented point. In
every case open a pull request and do not merge it; merging waits on Brando.

BRANCH
git checkout main && git pull && git checkout -b exp13-ginc-hmm

READ FIRST, IN THIS ORDER
1. experiments/13_ginc_hmm_complexity_vs_alignment/PROTOCOL.md: the design and the
   contract. Sections 2-5 must not change after your freeze commit.
2. experiments/ideas/complexity_vs_alignment/README.md: the program, the notation
   (epsilon excess loss in bits, T tokens seen, alpha target share, K exact
   complexity, eta, kappa, gamma0, gamma1), the law names (M0_scale, M_align_eff,
   M_align_add, M_K_coef, M_K_exp, M_K_both), claims C1-C3, and the shared rules.
3. experiments/ideas/predictability_bridge/README.md ("Where Kolmogorov complexity
   belongs") and experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md:
   never claim to measure universal Kolmogorov complexity.
4. experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/ (run.py,
   laws.py, independent_fit.py, analyze.py, verify.py, test_laws.py): the code
   layout, manifest and verification style to mirror.
5. experiments/13_ginc_hmm_complexity_vs_alignment/launch/ (README.md, snap.sh,
   sherlock.sbatch): how the GPU arm is launched. Your run.py must provide the
   command-line interface these scripts call (see run.py below).
6. If it exists (main or archive): Experiment 11's frozen PROTOCOL.md, Section 5
   (gate definitions G1-G4). If its G1-G4 differ from this protocol's, align this
   protocol's definitions BEFORE your freeze and say so in the freeze commit
   message and in CKPT.
7. Optional background: GINC, https://arxiv.org/abs/2111.02080, code
   https://github.com/p-lambda/incontext-learning (read-only; copy nothing). The
   diversity paper, https://arxiv.org/abs/2306.13840, Section 3.5 and Appendix K.

ENVIRONMENT (CPU box: 4 cores, 15 GB RAM, no GPU, no torch preinstalled)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
- Pin every installed version exactly in expt_v1/requirements.txt.
- In code, call torch.set_num_threads(1). Run at most 4 worker processes. Keep RAM
  under 12 GB.
- Record the torch version in config.json as "gpu_torch_version" (the GPU launch
  scripts install that version with CUDA).
- Write expt_v1/requirements-gpu.txt: the same pins minus torch (the scripts
  install torch first from the CUDA index).

FILES TO CREATE under experiments/13_ginc_hmm_complexity_vs_alignment/expt_v1/

config.json: every constant in the PROTOCOL, with arm-specific blocks "small"
and "gpu".
- Shared:
  - q 6; successors_per_state 3; emissions_per_state 4.
  - doc_len 128 plus BOS token id V.
  - alpha_grid [0.05, 0.15, 0.4, 1.0]; heldout_alpha 0.15.
  - Entropy band: entropy_band_bits 0.15; entropy_band_fallback 0.30;
    band_calibration_candidates 64; max_candidates_per_level 1000.
  - Slices: early [1,16], middle [17,64], late [65,128].
  - task_sampling_seed 20261013; eval_seed 20261020 (= 20261013 + 7);
    analysis_seed 29101326; bootstrap_draws 20000; n_starts 64.
  - Bounds: gamma (0,3]; gamma1 [-1,1]; eta [0,3]; kappa [-3,3]; beta (0,3].
  - rmse_reduction_threshold 0.10; bonferroni_alpha 0.025;
    g3_ratio 0.90; g3_gamma1_band 0.02; g5_share 0.5;
    g6_ratio 0.67; g6_reverse_ratio 1.5; fallback_fail_share 0.25.
  - independent_fit_tol 1e-4; verify_tol_bits 1e-9; prime 2**61-1.
- small:
  - V 64; levels [4,8,16,32]; heldout_level 16.
  - targets_per_level 2; distractors_per_level 2; pilot_concepts_per_level 1.
  - Model: layers 3, d_model 96, heads 4, mlp 384, vocab 65, tied embeddings;
    expected_params 354240; expected_nonembedding 335712.
  - Optimizer: lr 1e-3, betas [0.9,0.95], weight_decay 0.1, clip 1.0, warmup 32,
    constant schedule; batch_docs 8.
  - T_checkpoints [2^17..2^21]; T_checkpoints_fallback [2^16..2^20];
    fallback_if_projected_hours_above 3.0.
  - seeds [13101,13202,13303,13404]; pilot_seed 13909; pilot_alphas [0.05,0.4].
  - eval_docs_per_target 512; epsilon_star 0.05.
  - workers 4; threads 1; max_wall_seconds 14400.
- gpu:
  - V 150; levels [8,16,32,64,128]; heldout_level 32; targets 2, distractors 2,
    pilot 1 per level.
  - sizes S/M/L = (layers, d_model, heads):
      S (4, 256, 4)
      M (8, 512, 8)
      L (12, 768, 12)
    MLP width 4 d_model; vocab 151.
  - expected_params [3230976, 25362944, 85270272];
    expected_nonembedding [3159552, 25220096, 85056000].
  - lr [1e-3, 6e-4, 3e-4]; warmup 200; batch_docs 128; bf16 autocast for training.
  - T_checkpoints [2^22, 2^24, 2^26, 2^28, 2^30]; flop_cap 3e17 (6*N_total*T),
    which ends L at 2^29 with a final checkpoint at 2^29.
  - seeds [13101, 13202, 13303]; pilot_seed 13919; pilot_lr_multipliers
    [0.5, 1, 2]; pilot_T 2^26.
  - eval_docs_per_target 2048; epsilon_star 0.02; n_shards 360; data_workers 8.

hmm.py: the concept language and exact computations. Pure numpy plus Python
integers / fractions; no torch.
- sample_candidate(rng, n, V, q): successor sets and emission sets drawn without
  replacement, flat-Dirichlet probabilities, quantized to integer numerators
  >= 1 summing to 2**q by largest remainder after reserving one unit per entry.
- strongly_connected_aperiodic(h).
- process_rank(h): Krylov closure of forward vectors pi^T M_x... and backward
  vectors M_x... 1, with M_x = diag(O[:,x]) A, in integer arithmetic (scale by
  n * 2**(2q) per step) and Gaussian elimination mod p = 2**61 - 1. Return the
  minimum of the two span dimensions. If the result is < n and n <= 32, recheck
  with exact fractions.Fraction elimination. Accept only if rank == n.
- encode(h) / decode(bits) and K_bits(n, V, q) = n*(3*ceil(log2 n) +
  4*ceil(log2 V) + 5*q), exactly as PROTOCOL Section 2.2.
- forward_logprobs(h, docs): scaled forward algorithm in float64. Returns
  per-token code lengths -log2 p(x_t | x_<t) for a batch of documents.
- mixture_logprobs(concepts, prior, docs): exact Bayes-mixture code lengths. Run
  one forward pass per concept, then combine the per-concept prefix
  log-likelihoods with logsumexp at every position.
- entropy_rate(h, rng): PROTOCOL Section 2.3 estimator with its standard error.
- stationary(h), visit_tail_exponent(h), min_stationary(h),
  emission_overlap(h1, h2).
- build_concepts(cfg, arm): the acceptance procedure of PROTOCOL Section 2.1
  (band calibration on the first 64 rank-certified candidates per level, then
  acceptance in generation order). Order per level: 2 targets, 2 distractors,
  1 pilot. Log every candidate's outcome. Write tasks.json.
- Document stream: doc_stream(seed, schedule) yields token batches. The j-th
  document of concept c in training seed s comes from numpy PCG64 seeded by
  (s, c, j); the same (s, c, j) gives the same document in every run. Target slots
  follow the exact Bresenham rule: doc k is target iff
  floor((k+1)*alpha) - floor(k*alpha) == 1. Distractor slots cycle through a
  seeded permutation of the distractor pool.

model.py: the decoder-only transformer of PROTOCOL Section 2.6.
- Pre-LayerNorm, learned positions (128 slots: the input is BOS plus tokens
  1-127, predicting tokens 1-128), GELU, tied embeddings, no dropout, causal
  attention.
- param_count(model) and nonembedding_count(model). Non-embedding excludes the
  token and position embeddings.
- save_npz / load_npz: float32 numpy arrays keyed by parameter name, no pickle.

run.py: command-line interface.
  --arm small|gpu
  --check-only
  --pilot
  --output DIR (refuses to overwrite an existing non-empty DIR, except to resume
    shards in gpu mode)
  --shard I (gpu: run exactly one shard)
  --pilot --pilot-job J (gpu: run one of the 9 pilot jobs, J = 3*i_size + i_lr)
  --pilot-select DIR (gpu: read the 9 pilot jobs, apply the frozen rule, print
    and write the chosen multipliers)
  --merge DIR (gpu: collect shards into results/gpu)
  --workers 4
- Shard id = 120*i_N + 24*i_level + 12*i_target + 3*i_alpha + i_seed, matching
  PROTOCOL Section 7. Each shard writes OUT/shards/<%05d>/ and, last, a DONE file
  holding the SHA-256 of every file it wrote. Pilot job J writes
  OUT/pilot/<%05d>/ and its own DONE file the same way. A gpu shard or pilot job
  asserts at start that torch.cuda.is_available() and compute capability >= 8.0
  (bfloat16), and records the GPU name, driver and torch/CUDA versions in its
  manifest.
- --check-only does all of the following and trains nothing:
  - builds the concepts and prints, per level: candidates tried, the band used,
    accepted concepts with K, K_bits, process rank and entropy rate +- SE;
  - asserts the model parameter counts against config;
  - builds every run's alpha schedule and asserts exact target counts at every
    checkpoint;
  - hashes every eval document and asserts that none equals a training document
    of any run (compare the first 2^16 training documents of every concept and
    seed; documents are regenerated from seeds);
  - computes and caches the oracle and mixture code lengths (oracle.npz,
    mixture.npz);
  - prints the expected run count, the shard map (gpu) and a time estimate.
- --pilot runs the disclosed pilot of PROTOCOL Section 2.8 into results_pilot/
  (or OUT/pilot on GPU). It prints:
  - tokens/s per worker under 4-way load;
  - the projected full-run hours and the T range it implies;
  - the hardest fit level's epsilon_learn at the largest fit T, and the
    epsilon_star rule outcome;
  - (gpu) the chosen learning rate per size.
- Training:
  - same initialization for every run of one seed and size;
  - AdamW with weight decay on matrices only;
  - warmup then constant learning rate;
  - checkpoints at exactly the configured T in tokens.
- Evaluation at each checkpoint:
  - float64 forward over the eval documents with deterministic algorithms;
  - per-token model code lengths, then epsilon, epsilon_id and epsilon_learn per
    slice and the 128-position curve;
  - 64 fresh documents per distractor for the training-distribution loss.
- Writes metrics.csv, per_position.csv, mixes.json, checkpoints/...,
  manifest.json.
- Stops cleanly at max_wall_seconds with a resumable state and a log line.

laws.py:
- The six laws of PROTOCOL Section 4.1, each with the exact epsilon_id offset as
  given in the table there (M_align_add has none).
- The two full N laws (gpu) and their kappa = 0 and gamma1 = 0 variants.
- Fitting: scipy.optimize.least_squares with method "trf" on log2 epsilon,
  64-start grid, bounds from config, and gamma(K) > 0 enforced at every grid K.
- Invalid-forecast rule as in the PROTOCOL.
- T_star, M_K, M_alpha, the elasticities and epsilon_flip, all exactly as in
  PROTOCOL Section 4.4.

independent_fit.py:
- Refits every law with scipy.optimize.minimize (L-BFGS-B) on the same objective,
  from a different starting grid (seeded differently).
- Must agree with laws.py on every held-out RMSE to 1e-4 relative; prints AGREE
  or the first disagreement.

analyze.py:
- Per-seed fits on fit rows only. Forecasts for all held-out rows are written to
  forecasts.csv, and forecasts.csv is hashed into gates.json BEFORE any held-out
  outcome is read.
- Held-out RMSE, the paired seed-block bootstrap (20,000 draws, analysis seed),
  and the secondary two-stage (seed x target) bootstrap.
- Gates G1-G6 exactly as PROTOCOL Section 5 (G6 gpu only), with the per-seed
  sign rule for gpu.
- Position decomposition (Section 4.5).
- The epsilon_star curve, the sensitivity fits (no offset; K = n;
  K = K_bits - log2 n!), and the mechanism analyses (Section 4.6).
- Writes fits.json, gates.json, bootstrap.npz.

verify.py: every check in PROTOCOL Section 8. Prints PASS or the first failure.
- --arm small|gpu.
- --shard I --output DIR: verifies one gpu shard in DIR/shards/<%05d>/ and
  writes a VERIFIED marker next to DONE. The launch scripts use this in
  MODE=verify.
- Recompute model code lengths from checkpoints in float64 to max_abs_diff
  < 1e-9 bits.
- Recompute oracle and mixture code lengths with an independent log-space forward
  algorithm.
- Recertify process ranks and recompute K_bits.
- Regenerate training-stream hashes.
- Run independent_fit.py.

report.py: writes RESULTS_ginc_hmm.md and figures.
- Figures:
  - epsilon vs T per level and alpha (log-log, with the epsilon_id floor dashed);
  - the 128-position excess curves;
  - M_K and M_alpha vs epsilon;
  - early/late alignment shares r_alpha with intervals.
- Every gate appears in one table: estimate, 95% interval, threshold, verdict.

diversity.py (optional, no gate): if Experiment 14's code exists under
experiments/14_task2vec_vs_exact_complexity/expt_v1/ or its archive copy:
- import its diversity-coefficient function read-only;
- compute the diversity coefficient of each 13-small (target, alpha) training mix
  with Experiment 14's settings;
- write diversity.csv.
Otherwise skip, and write the follow-up line into RESULTS.

test_hmm.py, test_laws.py, test_model.py: the unit tests below.

requirements.txt, requirements-gpu.txt: exact pins.

UNIT TESTS THAT MUST PASS BEFORE ANY RUN
1. Forward algorithm: on random HMMs (n <= 4, V <= 4, length 6), the per-sequence
   probability equals brute-force summation over all hidden paths to 1e-12.
2. Process rank:
   - on tiny HMMs, the Krylov rank equals the exact Fraction rank of the
     brute-force Hankel matrix over all strings of length <= n-1 on each side;
   - a lumpable HMM (two states with identical emissions and outgoing rows) gets
     rank < n;
   - the mod-p and Fraction paths agree on 200 random n <= 8 cases.
3. Description length: decode(encode(h)) == h and len(encode(h)) == K_bits(n)
   for 1,000 random concepts. Also K_bits(4,8,16,32 | V=64) = 240, 504, 1056,
   2208 and K_bits(8..128 | V=150) = 568, 1184, 2464, 5120, 10624.
4. Entropy rate: on a visible Markov chain (each state emits a unique token), the
   estimate lies within 3 SE of -sum_i pi_i sum_j A_ij log2 A_ij.
5. Mixture:
   - at alpha = 1, mixture code lengths equal oracle code lengths exactly;
   - for every alpha, the total per-document regret (mixture minus oracle) is
     <= log2(1/alpha) + 1e-9 (a single document's regret may be negative), and its
     mean over 512 documents sampled from the target is >= 0 within 3 SE (it
     estimates a KL divergence).
6. Alpha schedule: exact floor(m*alpha) target documents after every prefix of m
   documents; distractor counts differ by at most 1.
7. Common random numbers: document j of concept c in seed s is identical across
   two different alpha runs.
8. Disjointness: no eval document hash equals any training document hash (on a
   reduced instance).
9. Model: parameter counts equal 354,240 / 335,712 for the small model, and the
   three GPU sizes match config. The save/load npz round trip is exact.
10. Laws:
    - each law fitter recovers known parameters from synthetic data with noise
      sd 0.02 (relative error < 5% on exponents);
    - independent_fit.py agrees to 1e-4;
    - the invalid-forecast rule triggers on a synthetic M_align_add forecast <= 0.
11. Position decomposition: on a synthetic balanced grid with known main effects,
    r_alpha is recovered to 1e-9.
If a test fails twice after honest fixes, write the blocker in CKPT_ginc_hmm.md
and stop.

ORDER OF WORK
1. Build the files. Run the unit tests (python -m pytest -q expt_v1). Run
   run.py --arm small --check-only and run.py --arm gpu --check-only (CPU is fine
   for the gpu check: it builds concepts and caches references, and trains
   nothing). Save both outputs into CKPT.
2. 13-small pilot: run.py --arm small --pilot (seed 13909, 4 pilot concepts,
   alpha in {0.05, 0.4}, 8 runs). Apply the two pre-declared rules:
   a. T range: 2^17..2^21 if the projected full run is <= 3.0 h, else 2^16..2^20.
   b. epsilon_star rule of PROTOCOL Section 4.4.
   Write a "Pilot disclosure" paragraph at the end of PROTOCOL Section 2 with the
   numbers and decisions. Seed 13909 is never used again.
3. If Experiment 11's frozen G1-G4 differ from these, align them now (step 6 of
   READ FIRST).
4. FREEZE: commit PROTOCOL.md, config.json and the code with the message
   "Freeze Experiment 13 protocol". No gated run may exist before this commit.
   Push the branch so the GPU launcher can check out this exact commit. Record the
   freeze SHA in CKPT.
5. 13-small full run under the wall cap:
     python expt_v1/run.py --arm small --output expt_v1/results/small --workers 4
   (128 runs).
6. verify.py --arm small must print PASS. If it fails, fix the cause, log it in
   CKPT, and rerun only what the fix touches. Every rerun is logged.
7. analyze.py --arm small, then report.py --arm small.
8. 13-GPU: you do not launch it. Write in CKPT the exact launch instructions from
   launch/README.md with the freeze SHA filled in. The GPU pilot (seed 13919,
   learning-rate check) is part of the GPU launch: MODE=pilot in the launch
   scripts runs the 9 pilot jobs (run.py --arm gpu --pilot --pilot-job J), then
   run.py --arm gpu --pilot-select <OUT>/pilot picks the rates. It runs at the
   freeze commit, before the shard array. Its chosen learning-rate multipliers, picked
   by the frozen rule of PROTOCOL Section 2.8, are recorded in ONE commit titled
   "Experiment 13 GPU pilot disclosure", made before any GPU shard. That commit
   changes only config.json key gpu.lr_selected and adds a "GPU pilot
   disclosure" paragraph to PROTOCOL Section 7; Sections 2-5 are untouched. The
   shard array then runs at that commit (its SHA is the TDC_COMMIT for launch).
   run.py --arm gpu --shard refuses to start if gpu.lr_selected is unset.
   - If results/gpu exists (merged by the person who ran it, via
     run.py --arm gpu --merge <OUT>): run verify.py --arm gpu, analyze.py and
     report.py for gpu, and include the GPU gates in RESULTS.
   - If not: RESULTS reports the 13-small gates and states "13-GPU: pending,
     not run". CKPT says the same. Skip step 11 (archive). Open the pull request
     as "Experiment 13 (13-small complete, 13-GPU pending)".
9. RESULTS_ginc_hmm.md (in the experiment folder) must contain:
   - every gate (G1-G5 per arm, G6 for gpu) with equal prominence, SUPPORTED or
     NOT_ESTABLISHED, each with its number and 95% interval;
   - the fitted laws per seed;
   - the multiplier curves;
   - the position decomposition, including epsilon_id vs epsilon_learn;
   - the epsilon_id-vs-alpha shape test (linear vs log(1/alpha));
   - the mechanism covariates (the tail exponent vs gamma correlation);
   - the sensitivity fits;
   - the diversity line;
   - the PROTOCOL Section 6 row that matches the outcome;
   - every pilot decision.
10. CKPT_ginc_hmm.md: a dated log (MM-DD-YYYY) of every step, pilot, failure,
    rerun, decision and timing. ARTIFACT_SHA256.json over every file in results/
    and the reports. GPU checkpoints stay on cluster storage: record their
    SHA-256 list (from the shards' DONE files) and do not commit them.
11. Archive (only when both arms are complete):
    - git mv experiments/13_ginc_hmm_complexity_vs_alignment
      experiments/archive/13_ginc_hmm_complexity_vs_alignment;
    - fix relative links that the move breaks;
    - add one row to experiments/README.md (setup and status, one line, numbers
      from RESULTS).
12. Push the branch and open a pull request with gh pr create.
    - Summary: at most 10 bullets, every gate verdict with its number.
    - Test plan: 2-3 bullets (unit tests, verify.py PASS, check-only).
    - Merging waits on Brando.

HARD RULES
- Never change a gate, threshold, estimator, seed, grid cell or count after the
  freeze commit. The only post-freeze edit allowed is the GPU pilot disclosure
  (config gpu.lr_selected plus a PROTOCOL Section 7 paragraph), by the frozen
  rule, in its own commit before any GPU shard.
- Never drop a seed, concept, run, checkpoint or forecast. Failed or diverged runs
  stay in, flagged, and count as invalid forecasts where they enter a law.
- No hidden reruns: every run, including failed ones, is in CKPT.
- If a unit test fails twice, stop and write the blocker in CKPT.
- Bounded resources on the CPU box: 4 workers x 1 thread, RAM < 12 GB,
  max_wall_seconds 14400, artifacts < 2 GB.
- Never run the GPU arm on the CPU box. Never open SSH connections in loops.
- No calls to any model-provider API. No credentials in the repository.
- Cite the alignment-scaling paper only as "the alignment-scaling paper with
  Varun", linking https://github.com/madanva/alignment-scaling-laws (no other link,
  no author list).
- No tool or AI attribution in commits, docs or the pull request.
- Plain English; expand acronyms on first use; dates MM-DD-YYYY.

DELIVERABLE CHECKLIST
[ ] expt_v1/ code, config.json, requirements.txt, requirements-gpu.txt, tests,
    all passing
[ ] run.py --check-only output (both arms) saved in CKPT
[ ] 13-small pilot disclosed in PROTOCOL.md; freeze commit made before any gated
    run; freeze SHA in CKPT
[ ] results/small complete: tasks.json, data/<task_id>_test.npz, oracle.npz,
    mixture.npz, mixes.json, metrics.csv, per_position.csv, checkpoints/, fits.json,
    forecasts.csv, gates.json, bootstrap.npz, manifest.json
[ ] verify.py --arm small prints PASS; independent_fit.py prints AGREE
[ ] results/gpu complete and verify.py --arm gpu PASS, OR "13-GPU: pending, not
    run" stated in RESULTS and CKPT with the launch instructions
[ ] RESULTS_ginc_hmm.md with every gate at equal prominence and the figures
[ ] CKPT_ginc_hmm.md (dated log)
[ ] ARTIFACT_SHA256.json
[ ] if both arms are complete: folder moved to
    experiments/archive/13_ginc_hmm_complexity_vs_alignment/, links fixed, row
    added to experiments/README.md
[ ] branch exp13-ginc-hmm pushed, pull request open, not merged
```

**TLDR-end:** [tdc: Expt 13 prompt] A self-contained coding-agent prompt that builds the certified-HMM concept generator, exact oracle and Bayes-mixture references, transformer runner and law fitters, then pilots, freezes, runs and verifies the CPU arm, reports gates G1–G5 at equal prominence, and either merges the GPU arm's sharded results or stops with GPU marked pending.
