# Coding-agent prompt: run Experiment 12 end to end

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/12_dsl_shortest_program_complexity_vs_alignment/PROMPT.md>

Paste the block below into Claude Code or Codex, running on the CPU box. It is self-contained.

```text
ROLE AND GOAL
You are a careful research engineer. Run Experiment 12 of the-difficulty-confound end
to end: build, test, pilot, freeze, run, verify, analyze, report, archive and open a
pull request.

The experiment trains a small transformer to compute 20 target functions from 10-bit
words to 10-bit words whose complexity is known exactly: the shortest program length
K in {2, 3, 4, 5, 6} in a fixed 12-primitive domain-specific language (DSL), found by
exhaustive breadth-first search (BFS) over distinct truth tables. Each target is
trained at target share alpha in {0.05, 0.15, 0.4, 1.0} of the training mix, against
a fixed pool of 20 distractor functions. It tests C1 (complexity costs data), C2
(alignment still dominates in units of data) and C3 (complexity moves the
scaling-law coefficient or the exponent). It also checks exact K against the Coding
Theorem Method / Block Decomposition Method (CTM/BDM) and compressors, and compares
its C3 verdict with Experiment 11.

The scientific design is
experiments/12_dsl_shortest_program_complexity_vs_alignment/PROTOCOL.md. It is the
authority. If anything below disagrees with it, follow PROTOCOL.md and log the
discrepancy in the CKPT file.

REPO AND BRANCH
  git clone https://github.com/brando90/the-difficulty-confound.git
  cd the-difficulty-confound
  git checkout main && git pull
  git checkout -b exp12-dsl-complexity-alignment
Work only inside experiments/12_dsl_shortest_program_complexity_vs_alignment/. The
exceptions are the archive move and the index rows at the end.

READ FIRST, IN THIS ORDER
 1. experiments/12_dsl_shortest_program_complexity_vs_alignment/PROTOCOL.md (all).
 2. experiments/ideas/complexity_vs_alignment/README.md: law names, claims C1-C3 and
    the shared rules.
 3. experiments/11_dfa_exact_complexity_vs_alignment/ (or its archive copy):
    PROTOCOL.md, PROMPT.md and, if it has run, expt_v1/*.py and results/analysis.json.
    Mirror its layout, CLI and analysis arithmetic. Copy (do not import) its law and
    bootstrap code, noting the source commit in a comment.
 4. Section 2 of experiments/14_task2vec_vs_exact_complexity/PROTOCOL.md and of
    experiments/15_alignment_alpha_recovery/PROTOCOL.md: they import dsl.py, data.py
    and model.py from here and read results/ (see CONSUMER CONTRACT).
 5. experiments/ideas/COLLABORATION_ONBOARDING.md, and
    experiments/archive/08_rlvr_verifiable_reward_utility_laws/ (PROTOCOL.md, CKPT,
    RESULTS, expt_v1/*.py): the workflow and code layout to mirror.
 6. experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md, and in
    experiments/ideas/predictability_bridge/README.md the sections "Three different
    variables" and "Where Kolmogorov complexity belongs". Never claim to measure
    universal Kolmogorov complexity. K here is exact only for this DSL.
 7. pybdm (https://github.com/sztal/pybdm), read-only; cite it in proxies.py.

ENVIRONMENT (CPU only: 4 cores, 15 GB RAM, no GPU, no torch or numpy preinstalled)
  python3 -m venv ~/venvs/exp12
  ~/venvs/exp12/bin/pip install --upgrade pip
  ~/venvs/exp12/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  ~/venvs/exp12/bin/pip install numpy scipy matplotlib pybdm
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Write expt_v1/requirements.txt with the exact installed versions (pip freeze) of
torch (+cpu build), numpy, scipy, matplotlib and pybdm, plus a comment with the torch
index URL. Use Experiment 11's torch and numpy pins if it has them (log any failure).
Every worker calls torch.set_num_threads(1). Use ~/venvs/exp12/bin/python throughout.
pybdm 0.1.0 worked with numpy 2.5.4 on 10-18-2026. If it fails to install or import,
follow the fallback in PROTOCOL Section 3.2.

FILES TO CREATE under experiments/12_dsl_shortest_program_complexity_vs_alignment/expt_v1/

config.json
  Every constant, with no magic numbers in code:
  - seeds [12101,12202,12303,12404,12505,12606,12707,12808]; pilot_seed 12909;
    task_seed 20261012; analysis_seed 29101226; bootstrap_draws 20000.
  - word_bits 10; primitives in index order ["NOT","REV","ROTL1","ROTR1","SHL1",
    "SHR1","GRAY","PREFIX_XOR","SWAP_HALVES","SWAP_ADJ","INC","NEG"]; max_K 6;
    bfs_chunk_parents 8192; bfs_ram_limit_gb 4.
  - K_levels [2,3,4,5,6]; K_bits [11,14,20,23,27]; K_bits_center 18.04;
    len_to_bits_slope 0.842; 4 targets, 4 distractors per level; 4 probe and 4 fresh
    per stratum; selection_chunk 4096; balance_range [0.3,0.7]; min_depends 8;
    strata [[0.08,0.11],[0.11,0.14],[0.14,0.22],[0.30,1.0]] (half-open; S4 closed at
    1); cell_types ["affine","nonaffine","affine","nonaffine"]; rare_rule_count 16.
  - n_train 512; n_test 512; train_bit_count_range [236,276]; alphas
    [0.05,0.15,0.4,1.0]; fit_K [2,3,5,6]; heldout_K [4]; fit_alphas [0.05,0.4,1.0];
    heldout_alphas [0.15]; T_checkpoints [1024,...,65536] (7); fit_T_max 32768.
  - batch_size 64; lr 1e-3; weight_decay 0.01; betas [0.9,0.999]; grad_clip 1.0;
    warmup_steps 8; d_model 64; n_layers 2; n_heads 4; d_mlp 256; max_positions 11;
    n_task_tokens 40; expected_parameters 103618 (103682 with the null token).
  - eps_star_primary 1.0; eps_star_levels [3.0,1.0,0.3]; eps_res 1e-2 (bits per
    example); plateau 0.8 and 1.0 bits; memorize_train_bits 0.1; not_fit 1.0.
  - pilot: T_max 131072; P1 16384; P2 5 targets at T 2048; P3 0.9; P4 10 targets.
  - gates: g1 2.0, lb 1.0; g2 0.10; g3b 0.05; g4 1.0, mf_censor_fallback 0.5 at
    3.0 bits; g5 0.5, ub 1.0; g6 0.10; g7 -0.5; g8 0.5; g9 [0.5,2.0].
  - workers 4; threads 1; max_wall_seconds 18000; artifact_limit_gb 2.0; rep768
    (n_train 768, alphas [1.0,0.15], first 4 seeds, 60 min, note ratio 0.5); mlp
    (hidden [256,256], 83988 parameters, alpha 1, first 4 seeds, 10 min); bdm (block
    12, ndim 1, "ignore"); zlib 9; lzma 9; Experiment 11 analysis.json paths (live
    and archive); fit start grids (PROTOCOL Section 4.1).

dsl.py  (public API: Experiments 14 and 15 import it, so keep these names)
  - PRIM: (12, 1024) uint16 lookup tables in index order, from the bit formulas in
    PROTOCOL Section 1. apply_program(program): truth table of a sequence of names
    or indices applied left to right (x -> p_K(...p_1(x))). k_bits(K).
  - bfs_enumerate(config) -> tables (uint16, (N, 1024)), K, canonical (int8, (N, 6),
    -1 padded), n_shortest, level_counts, seconds, peak_rss_gb. Exact dedupe: 64-bit
    hash, then full-row comparison within equal hashes; raise on a true collision.
    Parents in chunks of bfs_chunk_parents; child order (parent order, primitive
    index), so the first hit is the lexicographically smallest shortest program.
    n_shortest sums parents' counts over the (parent, primitive) pairs that hit each
    new function (numpy unique/bincount). Cache to results/bfs_cache.npz (git
    ignored). Assert peak RSS < bfs_ram_limit_gb.
  - exact_k(table, bfs) by lookup. brute_force(max_len) for tests and --check-only.
  - features(table): output_balance, output_bit_balance (10), n_depends, sensitivity,
    is_affine, local_rule_entropy, n_local_rules, rare_rule_mass. The local rule J(x)
    is the 10x10 bit matrix with (i, j) = 1 iff flipping input bit i flips output
    bit j, packed into a 100-bit key. fourier_sensitivity(table) for the test only.
  - train_identifiable(table, K, bfs, train_idx): no other function with K' <= K has
    the same restriction to train_idx.
  - select_tasks(config, bfs, train_idx): PROTOCOL Section 2.4 exactly. Per K, rng =
    numpy default_rng([task_seed, K]) shuffles that level, scanned in chunks of
    selection_chunk. Filters in the order balance, depends, identifiability,
    distinctness (including complements); count rejections by reason and level.
    Returns 200 task dicts with every tasks.json field (PROTOCOL Section 8). Ids
    dsl_k2_t1..t4, dsl_k2_d1..d4, dsl_k2_p01..p16, dsl_k2_f01..f16 in stratum order.
    Task tokens 0..19 to targets (level, then stratum order), 20..39 to distractors,
    null for probe and fresh tasks.

proxies.py
  bdm_rowmajor, bdm_bitplane (pybdm on the 10,240-bit strings of PROTOCOL Section
  3.2), zlib_bytes, lzma_bytes (np.packbits of the row-major bits), zlib_u16_bytes,
  and block_entropy_fallback (12-bit blocks, labelled "substitute", used only if pybdm
  is unavailable).

data.py  (public API)
  - to_bits(x) -> uint8 (n, 10) with bit i in column i; from_bits.
  - make_split(config) -> perm, train_idx = perm[:n_train], test_idx =
    perm[n_train:], attempts. perm comes from default_rng([task_seed, attempt]), with
    attempt incremented until every input bit is 1 in 236..276 of perm[:512]. The
    rep768 arm uses perm[:768] and perm[768:] of the same perm.
  - build_data(config, tasks, split, out_dir): data/<task_id>_{train,test}.npz for all
    200 tasks with x, y (uint16), bits_x, bits_y (uint8), in split order.
  - assemble_mix(config, tasks, target_id, alpha, seed, n_train=512) returns
    per-example arrays (task_index, input_index) for T_max = 65536:
      - in each octave interval [1,1024], (1024,2048], ..., (32768,65536], exactly
        round(alpha * interval_length) target slots, placed uniformly at random;
      - the other slots dealt to the 20 distractors in equal counts (+-1);
      - each task reads its training inputs in successive passes, each pass a fresh
        permutation from the data-order rng default_rng([seed, target_token,
        round(100*alpha)]).
    It also returns per-checkpoint per-task counts and target passes for mixes.json.
    Experiment 15 imports it to rebuild every mix: pure and deterministic.
  - example_stream(config, task, n, seed, partition="train"): the first n examples of
    successive permutation passes over the partition, rng default_rng([seed, h]) with
    h = the first 4 bytes of SHA-256(task_id) as an unsigned integer. For consumers
    that need more than 512 examples per task; content repeats after 512.
  - sample_inputs(rng, n, partition, split): uniform draws with replacement.

model.py  (public API)
  - build_model(config, n_task_tokens=40): bidirectional transformer (no causal
    mask), pre-LayerNorm, GELU, learned positions (11). Modules by name: tok_emb
    (vocab 2 + n_task_tokens), pos_emb, blocks.{0,1}.{ln1, qkv, out, ln2, fc1, fc2},
    ln_f, head = Linear(64, 2). qkv is one Linear(64, 192): output rows 0:64 query,
    64:128 key, 128:192 value, each head-major (4 heads x 16); attention via
    torch.nn.functional.scaled_dot_product_attention. Document this layout in the
    docstring (Experiment 14 rescales query, key and value by slicing it). No weight
    tying. Input [task token, x_0..x_9], token id 2 + t for task token t; logits
    (B, 10, 2) at positions 1..10. Assert 103,618 parameters at 40 tokens and
    103,682 at 41.
  - build_mlp(config, n_task_tokens=40): one-hot task plus 10 bits, then 256, 256
    (GELU), then 20 logits as (B, 10, 2). 83,988 parameters.
  - export_npz / load_npz: float32 arrays keyed by state_dict name, no pickle.
  - evaluate(model, bits_x, task_token, bits_y): one batch in the given order,
    float32 forward, per-example sum over 10 bits of -log2 p(correct), accumulated in
    float64. Returns mean bits, accuracy (all 10 bits), bit accuracy, per-output-bit
    bits and per-example bits. run.py and verify.py both call exactly this.

run.py
  - train_cell(config, tasks, target_id, alpha, seed, out_dir, n_task_tokens=40,
    arch="transformer", n_train=512, save_checkpoints=True) is public (Experiment 15
    imports it). torch.manual_seed(seed) before the model is built, so all cells of a
    seed share an initialization. AdamW, linear warmup over 8 steps, then constant.
    Loss: sum of 10 bitwise cross-entropies, batch mean. At each T in T_checkpoints,
    evaluate test and train sets and (if save_checkpoints) write
    checkpoints/<seed>_<target_id>_<alpha:.2f>/T_<T>.npz. Write metrics rows,
    predictions npz and checkpoints atomically (temporary name, then rename), then a
    done marker.
  - CLI: --check-only (BFS, selection, split, data, covariates, proxies and every
    invariant check below; one JSON receipt; no training). --pilot (seed 12909,
    alpha = 1, all 20 targets, T to 131072, into results_pilot/; prints the P1-P4
    verdicts). --output DIR (default results/; refuses a non-empty DIR unless
    --resume). --resume (skips done cells; allowed once; logged). --rep768 and --mlp
    (secondary arms after the main grid, each only if its min_minutes_left remains;
    predictions only). --only ARM:SEED:TARGET:ALPHA --output SCRATCH (one cell for
    determinism checks; never writes into results/).
  - Run the 640 cells with ProcessPoolExecutor(max_workers=4), 1 torch thread each.
    Free the BFS arrays first; workers get only the 40 trained truth tables. Stop
    scheduling at max_wall_seconds and record missing cells. If projected disk use
    exceeds artifact_limit_gb, stop and ask Brando.
  - Shared output contract in results/: tasks.json (200 tasks), data/, mixes.json,
    metrics.csv (columns arm, seed, target_id, K, K_bits, alpha, T, step,
    excess_bits, accuracy, bit_accuracy, n_test, excess_bits_by_output_bit (JSON),
    train_bits, train_accuracy, target_examples_seen, distractor_examples_seen,
    target_passes, wall_seconds), predictions/, checkpoints/, bfs_counts.json, and
    manifest.json (SHA-256 of every source and config.json, library versions, host,
    CPU model, UTC start and end, the check receipt, pilot summary, split attempts,
    any pybdm fallback).
  - The --check-only receipt includes: BFS level counts (must be 12, 112, 988, 8543,
    73146, 622876), seconds and peak RSS; brute force for K <= 4 (9,656 functions;
    same K, canonical program and n_shortest); rejections by filter and level,
    eligible counts per (stratum, type) cell, and identifiability clashes (shorter
    and equal length); the task table (id, role, K, K_bits, program, balance,
    sensitivity, stratum, affine, H_rule, n_shortest, BDM, zlib, lzma); pairwise
    distinctness and non-complementarity of all 200 truth tables; the split's
    per-bit train counts; a schedule check for one cell per alpha; parameter counts
    (103,618; 103,682; 83,988); the G8a statistic with its task-bootstrap interval;
    a 20-step timing probe with 4 concurrent workers; projected hours and gigabytes.

laws.py
  - LAWS: M0_scale, M_align_eff, M_align_add, M_K_coef, M_K_exp, M_K_both, M_K_floor,
    plus the per-target power law. K regressor selectable: K_bits (primary) or
    program length. Fit internally on log2(K / center); report gamma0 uncentered.
  - Censored residual on log2 epsilon at eps_res = 1e-2 (PROTOCOL Section 4.1).
  - scipy.optimize.least_squares (trf), the bounds and the 64-start grid. The
    exponent bound applies to gamma0 + gamma1*log2 K at K_bits 11 and 27.
  - M_align_add is fit on epsilon in linear space; forecasts <= 0 are invalid.
  - predict(); t_star(params, K, alpha, eps_star); elasticities(params, K_center,
    eps_star) with e_alpha = eta and analytic e_K (PROTOCOL Section 4.4).

independent_fit.py
  Profile log A out in closed form, then L-BFGS-B over the exponents from the same
  starts. Objective and held-out forecasts must agree with laws.py to 1e-4 relative.

analyze.py
  Stage 1 reads only fit rows of the main arm (filtered at read time), fits every
  law per seed on the 288 fit points, picks the G2 K law by fit-cell RMSE, writes
  fits.json and forecasts_frozen.json (272 points per seed) and prints its SHA-256
  (paste it into CKPT). Stage 2 reads held-out rows and computes everything in
  PROTOCOL Sections 4.2-4.6: RMSE per law and slice (K = 4, alpha = 0.15,
  T = 65536); tstar.csv with censor flags, in examples and passes; M_K (seed-block
  and two-stage) and M_K by type; E in both units, E_mf, the E(eps*) curve and
  eps_flip; R; eta and M_align_eff versus M_align_add; eta_j versus K and primitive
  overlap; the G5 gap ratio; G6; gamma_j and its Spearman with H_rule (all 20, the
  10 non-affine, and partial on log K_bits); the T* regression on log K_bits,
  sensitivity and affine, with variance inflation factors; the simplicity-bias
  table (K, both BDM orders, zlib, lzma, partials); the train-fit census by level
  and type; per-output-bit loss; the rep768 and MLP arms.
  Bootstrap: 20,000 draws resampling seeds (analysis seed 29101226); the two-stage
  version also resamples the 4 targets within each level; G8a resamples tasks
  within each level; percentile 95% intervals. G9 reads Experiment 11's
  analysis.json (SHA-256 logged); if absent, G9 is NOT_ESTABLISHED ("Experiment 11
  unavailable"). Evaluate G1-G9 exactly as PROTOCOL Section 5. Write analysis.json.

verify.py
  In order: (1) source and config hashes against manifest.json; (2) BFS rerun from
  scratch, level counts, and every task's K, canonical program and n_shortest, with
  each canonical program reproducing its truth table; (3) every task regenerated
  from the task seed, with every filter, covariate and proxy recomputed; (4) split,
  data/ and mixes.json regenerated byte-identical; (5) every main-grid checkpoint
  reloaded via load_npz, evaluate() rerun on test and train sets, and every
  metrics.csv excess_bits, train_bits and accuracy matched with max_abs_diff < 1e-9
  bits; (6) predictions matched; (7) 8 secondary-arm cells (chosen with the analysis
  seed) retrained with --only into scratch, bitwise-equal predictions; (8) T*
  recomputed; (9) both fitters rerun, agreement within 1e-4; (10) every gate
  recomputed and compared to analysis.json. Writes results/verification.json with
  status PASS or FAIL and the counts.

report.py
  Writes RESULTS_dsl_complexity_alignment.md in the experiment folder and a figure
  (dsl_complexity_alignment_results.png and .pdf) with four panels: (a) alpha = 1
  learning curves by K (seed mean and band; affine dashed); (b) T* heatmap over
  K x alpha at the primary eps*; (c) E versus eps* with E = 1 and eps_flip;
  (d) held-out forecasts versus observed for the G2 K law and M_align_eff. A second
  figure plots per-target T* against K, BDM and zlib.

test_dsl.py, test_data.py, test_model.py, test_laws.py (see UNIT TESTS)
.gitignore (experiment-local): results/checkpoints/, results_pilot/checkpoints/,
results/bfs_cache.npz, *.tar

UNIT TESTS (all must pass before --check-only; run each with the venv python)
test_dsl.py
  - Each primitive equals a plain per-bit Python loop on all 1,024 inputs.
  - GRAY o PREFIX_XOR = PREFIX_XOR o GRAY = identity; NEG = INC o NOT; NOT, REV,
    SWAP_HALVES and SWAP_ADJ are involutions; ROTL1 o ROTR1 = identity.
  - Hand values: sensitivity 0.1 for NOT, REV and SWAP_HALVES, 0.19 for GRAY, 0.55
    for PREFIX_XOR, 0.1396 (4 decimals) for INC and NEG; SHL1 depends on 9 bits and
    has overall balance 0.45; k_bits gives 5, 11, 14, 20, 23, 27 for K = 1..6.
  - BFS level counts 12, 112, 988, 8543, 73146, 622876.
  - Brute force over all 271,453 programs of length <= 5 (including the empty one):
    the same functions, K, canonical program (the lexicographic minimum of the
    shortest programs) and n_shortest as BFS for every K <= 5 function.
  - sensitivity equals fourier_sensitivity to 1e-12 on 200 random tables.
  - Every function whose canonical program has no INC or NEG is affine; is_affine
    matches a direct GF(2) linearity check on all K <= 4 functions and 200 random
    tables.
  - Local-rule statistics match an independent per-input Python loop on 50 tables;
    H_rule = 0 for every affine table.
  - select_tasks: counts per role and cell, the type pattern, distinct truth tables
    and complements, train-identifiability of targets and distractors, and
    identical output on a rerun.
test_data.py
  - Train and test inputs are disjoint and cover 0..1023; labels equal
    apply_program(canonical program) everywhere.
  - assemble_mix: exact target count per octave; distractor counts equal within +-1;
    per-input repeat counts are floor or ceil of n/512 at every checkpoint; same
    seed gives identical output, another seed differs.
  - Token-weighted alpha (Eq. 2 of the alignment-scaling paper with Varun) equals
    example-weighted alpha exactly.
  - example_stream: the first 512 are a permutation of the partition; deterministic.
test_model.py
  - Parameter counts 103,618 (40 tokens), 103,682 (41 tokens), 83,988 (MLP).
  - The documented qkv layout is right: scaling the query rows (weights and biases)
    by 2 and the key rows by 1/2 leaves the logits unchanged to 1e-5.
  - Changing input bit 9 can change the logits at position 1 (no causal mask).
  - An npz round trip gives identical logits; two 20-step train_cell runs with the
    same seed give bitwise-identical weights and a different seed does not;
    evaluate() equals a naive per-example loop to 1e-12.
test_laws.py
  - Synthetic data on the real design grid (K_bits 11..27, 8 synthetic seeds,
    log-normal noise sd 0.05) from M_K_coef, M_K_exp and M_K_both with known
    parameters: recovery within 0.05 absolute for eta, gamma, gamma0, gamma1, kappa.
  - G3 logic returns G3a on coefficient-generated and G3b on exponent-generated data.
  - The independent fitter agrees within 1e-4; the censored residual is zero when
    forecast and observation are both below eps_res; t_star is exact on a pure power
    law with correct censor flags; analytic e_K equals the finite difference within
    1e-6; M_align_add negative forecasts are flagged invalid; a bootstrap over
    identical seeds collapses to the point estimate.
  - G9 on mocked Experiment 11 outputs: same verdict and sign with ratio 1.5 gives
    SUPPORTED; ratio 2.5, opposite sign or unresolved Experiment 11 gives
    NOT_ESTABLISHED.

CONSUMER CONTRACT (Experiments 14 and 15 depend on this; do not break it)
  tasks.json has 200 tasks (roles target, distractor, probe_pretrain, fresh; full
  truth tables; family "dsl"). dsl.py, data.py and model.py import with no side
  effects, hashed in manifest.json; build_model(n_task_tokens=41) works and head is
  a separate module. mixes.json plus assemble_mix rebuild every mix exactly.
  Training files hold 512 rows per task and every mix uses all of them, so consumers
  needing 2,048 or 4,096 examples, or references outside the mixes, use
  example_stream or sample_inputs, and content repeats (say so in RESULTS).

ORDER OF WORK
 1. Branch, read, set up the environment.
 2. Write the code and tests. Run every test file. Commit ("Experiment 12: code and
    tests").
 3. run.py --check-only. Save the receipt to results_check.json. Commit it.
 4. Pilot (disclosed). run.py --pilot. Apply PROTOCOL Section 2.8 rules P1-P4
    mechanically. Write a "Pilot disclosure" paragraph into PROTOCOL Section 2.8:
    per-target T*(1.0) at alpha = 1, train versus test bits, step time, projected
    hours and gigabytes, any window or eps* change, how many targets improve from
    2^16 to 2^17, and the G8a value from the receipt. Seed 12909 is never reused;
    pilot outputs are never pooled. If P3 or P4 fires, stop and ask Brando.
 5. Freeze. In PROTOCOL.md, replace the Status line with "Frozen on MM-DD-YYYY before
    any gated Experiment 12 training." Apply only pilot-rule changes. Commit
    PROTOCOL.md (and config.json if P1 or P2 fired) alone as "Freeze Experiment 12
    protocol". Put that commit hash in CKPT. From now on, Sections 2-5 of
    PROTOCOL.md and every gate value in config.json are read-only.
 6. Full run, detached (nohup ~/venvs/exp12/bin/python expt_v1/run.py --output
    expt_v1/results > run.log 2>&1 &, or a named tmux session). Log start and end in
    CKPT. Then --rep768 and --mlp, each only if its margin remains.
 7. verify.py must print PASS. On FAIL, do not edit outputs. Diagnose, fix only
    verifier bugs, log each step in CKPT, and rerun verify.
 8. analyze.py (stage 1, paste the hash into CKPT, then stage 2), then report.py.
 9. Write CKPT_dsl_complexity_alignment.md, a dated log (MM-DD-YYYY) of every step:
    commands, hashes, wall times, deviations, failures. Do not tar the checkpoints
    (a tar would double disk use past 2 GB). Write ARTIFACT_SHA256.json covering
    every results file, including every checkpoint.
10. Archive: git mv experiments/12_dsl_shortest_program_complexity_vs_alignment
    experiments/archive/12_dsl_shortest_program_complexity_vs_alignment. Add a row
    to experiments/README.md and experiments/archive/README.md. Point the Experiment
    12 link in experiments/ideas/complexity_vs_alignment/README.md at the archive
    path. Checkpoints stay on disk, uncommitted (Experiments 14 and 15 run on this
    box and read them in place).
11. git push -u origin exp12-dsl-complexity-alignment, then gh pr create:
      title "Experiment 12: exact DSL program complexity versus alignment";
      body: Summary (at most 10 bullets, every gate with its verdict) and
      Test plan (2-3 bullets: unit tests, verify.py PASS, check-only receipt).
    Do not merge. Merging waits on Brando.

RESULTS FILE MUST CONTAIN
  - A gate table (gate, statistic, estimate, 95% intervals, threshold, verdict) for
    G1-G9, with G3a/G3b and G8a/G8b as separate rows, all with equal prominence.
  - The target table: id, K, K_bits, canonical program, sensitivity, stratum, affine,
    H_rule, BDM, zlib, T* per eps* (examples and passes).
  - Held-out RMSE per law and slice; eta and its reading (transfer, pure effective
    data or interference); E at 3.0/1.0/0.3 bits in both units; eps_flip; R; the
    K-versus-sensitivity-versus-type regression; M_K by type; the simplicity-bias
    table, stating the CTM/BDM-versus-K result plainly (the prototype predicted a
    negative correlation); the train-fit census; the rep768 and MLP arms (if run).
  - Pilot disclosure, deviations, limitations (PROTOCOL Section 9), the 512-row
    consumer note, the artifact location. Say "K is exact for this 12-primitive
    DSL", never "Kolmogorov complexity measured".

HARD RULES
  - Never change a gate, threshold, split, seed list, task list or eps* after the
    freeze commit. Never drop a seed, task, checkpoint or forecast; censored and
    plateaued runs stay in, with flags. No hidden reruns: every run, resume, crash
    and retry is logged in CKPT.
  - If a unit test or check fails twice for the same cause after a fix attempt, stop.
    Write the blocker in CKPT, push the branch, and open a draft pull request that
    describes it.
  - Bounded resources: 4 workers, 1 thread each, RAM under 12 GB (under 4 GB during
    BFS), artifacts under 2 GB, wall cap 18,000 s.
  - No model-provider API calls. No credentials, tokens or keys in the repo or logs.
    No AI or tool attribution in commits, files or the pull request.
  - Plain English. Expand acronyms on first use. Dates MM-DD-YYYY. End every new
    markdown file with a "**TLDR-end:** [tdc: ...]" line.

DELIVERABLE CHECKLIST
  [ ] expt_v1/: config.json, dsl.py, proxies.py, data.py, model.py, run.py, laws.py,
      independent_fit.py, analyze.py, verify.py, report.py, test_dsl.py,
      test_data.py, test_model.py, test_laws.py, requirements.txt, .gitignore
  [ ] all four test files pass; check-only receipt committed
  [ ] pilot disclosed in PROTOCOL Section 2.8 and CKPT; seed 12909 not reused
  [ ] freeze commit of PROTOCOL.md before any gated run; hash in CKPT
  [ ] 640/640 cells complete (or the wall-cap shortfall logged), within 18,000 s
  [ ] results/: tasks.json (200 tasks), data/ (400 files), mixes.json, metrics.csv
      (4,480 main rows plus secondary-arm rows), predictions/, checkpoints/ (on
      disk), bfs_counts.json, tstar.csv, fits.json, forecasts_frozen.json,
      analysis.json, manifest.json, verification.json (PASS)
  [ ] RESULTS_dsl_complexity_alignment.md with G1-G9 and both figures (png, pdf)
  [ ] CKPT_dsl_complexity_alignment.md dated log
  [ ] ARTIFACT_SHA256.json (including every checkpoint file)
  [ ] folder moved to experiments/archive/; rows in experiments/README.md and
      archive/README.md; overview link updated
  [ ] branch exp12-dsl-complexity-alignment pushed; pull request open, not merged
```

**TLDR-end:** [tdc: Expt 12 prompt] A self-contained coding-agent prompt that builds the exact breadth-first search over a 12-primitive bit-word language, the stratified task selector, mix assembly, a 103,618-parameter bidirectional transformer, CTM/BDM and compressor proxies, and the law fitters. It tests them against brute force and synthetic data, runs a disclosed pilot, freezes the protocol, trains 640 CPU runs under a 5-hour cap, replays every checkpoint, and reports gates G1–G9 in a pull request.
