# Coding-agent prompt: run Experiment 11 end to end

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/11_dfa_exact_complexity_vs_alignment/PROMPT.md>

Paste the block below into Claude Code or Codex, running on the CPU box. It is self-contained.

```text
ROLE AND GOAL
You are a careful research engineer. Run Experiment 11 of the-difficulty-confound end
to end: build, test, pilot, freeze, run, verify, analyze, report, archive and open a
pull request.

The experiment trains a small transformer to accept or reject binary strings for 16
target regular languages whose complexity is known exactly: minimal deterministic
finite automaton (DFA) states n in {4, 8, 16, 32}, found by Hopcroft's algorithm.
Each target is trained at target share alpha in {0.05, 0.15, 0.4, 1.0} of the training
mix, against a fixed pool of 16 distractor DFAs. It tests three claims:
  C1  complexity costs data;
  C2  alignment still dominates in units of data;
  C3  complexity moves the scaling-law coefficient or the exponent.

The scientific design is experiments/11_dfa_exact_complexity_vs_alignment/PROTOCOL.md.
It is the authority. If anything below disagrees with it, follow PROTOCOL.md and log
the discrepancy in the CKPT file.

REPO AND BRANCH
  git clone https://github.com/brando90/the-difficulty-confound.git
  cd the-difficulty-confound
  git checkout main && git pull
  git checkout -b exp11-dfa-complexity-alignment
Work only inside experiments/11_dfa_exact_complexity_vs_alignment/. The exceptions are
the archive move and the index rows at the end.

READ FIRST, IN THIS ORDER
 1. experiments/11_dfa_exact_complexity_vs_alignment/PROTOCOL.md (all of it).
 2. experiments/ideas/complexity_vs_alignment/README.md: law names, claims C1-C3 and
    the shared rules.
 3. experiments/ideas/COLLABORATION_ONBOARDING.md: the workflow (freeze, --check-only,
    verify.py replay, seed-block bootstrap, retain counterevidence).
 4. experiments/archive/08_rlvr_verifiable_reward_utility_laws/: PROTOCOL.md, CKPT,
    RESULTS, and expt_v1/*.py. Mirror its layout: config.json, run.py with
    --check-only, laws.py, independent_fit.py, analyze.py, verify.py, report.py,
    test_laws.py.
 5. experiments/archive/09_headroom_bounded_rlvr_utility_law/PROTOCOL.md, if it exists. It is
    the model for a frozen protocol with a disclosed pilot.
 6. experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md, and in
    experiments/ideas/predictability_bridge/README.md the sections "Three different
    variables" and "Where Kolmogorov complexity belongs". Never claim to measure
    universal Kolmogorov complexity. K here is exact only for the DFA description
    language.

ENVIRONMENT (CPU only: 4 cores, 15 GB RAM, no GPU, no torch or numpy preinstalled)
  python3 -m venv ~/venvs/exp11
  ~/venvs/exp11/bin/pip install --upgrade pip
  ~/venvs/exp11/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  ~/venvs/exp11/bin/pip install numpy scipy matplotlib
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Write expt_v1/requirements.txt with the exact installed versions (from pip freeze) of
torch (+cpu build), numpy, scipy and matplotlib. Add a comment line with the torch index
URL. Every worker calls torch.set_num_threads(1). Use ~/venvs/exp11/bin/python for
everything.

FILES TO CREATE under experiments/11_dfa_exact_complexity_vs_alignment/expt_v1/

config.json
  Every constant, with no magic numbers in code:
  - seeds [11101,11202,11303,11404,11505,11606,11707,11808]; pilot_seed 11909;
    task_seed 20261011; analysis_seed 29101126; bootstrap_draws 20000.
  - n_levels [4,8,16,32]; targets_per_level 4; distractors_per_level 4;
    candidates_per_level 64; sampler_overprovision 1.25.
  - accept_range [0.35,0.65]; per_length_accept_range [0.10,0.90].
  - alphabet ["a","b"]; length_min 8; length_max 24; test_hash_byte_below 32;
    test_size 4096; pool_per_length 4096.
  - alphas [0.05,0.15,0.4,1.0]; fit_n [4,8,32]; heldout_n [16]; fit_alphas
    [0.05,0.4,1.0]; heldout_alphas [0.15].
  - T_checkpoints [512,1024,...,65536]; fit_T_max 32768.
  - batch_size 64; lr 1e-3; weight_decay 0.01; betas [0.9,0.999]; grad_clip 1.0;
    warmup_steps 8.
  - d_model 64; n_layers 2; n_heads 4; d_mlp 256; max_positions 25;
    expected_parameters 104002.
  - eps_star_primary 0.1; eps_star_levels [0.3,0.1,0.03]; eps_res 1e-3.
  - gate thresholds:
      g1_min_multiplier 2.0; g1_lb_above 1.0;
      g2_min_rel_gain 0.10;
      g3b_min_rel_gain 0.05;
      g4_min_E 1.0; g4_mf_censor_fallback 0.5;
      g5_max_ratio 0.5; g5_ub_below 1.0;
      g6_min_rel_gain 0.10;
      g7_min_spearman 0.5.
  - workers 4; threads 1; max_wall_seconds 18000.
  - lstm arm: embed 32, hidden 128, expected_parameters 84290, min_minutes_left 45.
  - fit start grids (PROTOCOL Section 4.1).

dfa.py  (public API: Experiments 14 and 15 import it, so keep these names)
  - DFA representation: transitions int array of shape (n, 2), accept bool (n,),
    start 0.
  - sample_minimal_dfa(rng, n): draw m = ceil(1.25 n) + 1 states with uniform
    transitions and Bernoulli(0.5) accept bits; keep the reachable part; minimize;
    accept iff the result has exactly n states; relabel in breadth-first canonical
    order.
  - hopcroft_minimize(dfa) is the production algorithm. moore_minimize(dfa) is the
    brute-force partition-refinement reference.
  - canonical_form(dfa) for language identity. equivalent(d1, d2) by product-automaton
    BFS. complement(dfa).
  - run_dfa(dfa, bits, lengths): vectorized numpy labels.
  - Exact dynamic programs over the input distribution (length uniform on [8,24],
    uniform characters):
      acceptance_rate and acceptance_by_length;
      sensitivity, by propagating (original, substituted) state pairs;
      visit_freq, the expected visits over positions 0..l averaged over l, normalized;
      visit_tail_exponent b, the OLS slope of log freq on log rank.
  - select_tasks(config):
      For each n: draw candidates with rng seeded from (task_seed, n). Keep those that
      pass both acceptance filters and are neither equal to nor the complement of any
      kept candidate or earlier task, until there are 64.
      Sort by sensitivity and cut into 4 quartile bins of 16.
      From each bin, draw 2 without replacement: the first is the target, the second
      the distractor.
      Task ids: dfa_n04_t1..t4 and dfa_n04_d1..d4 (similarly n08, n16, n32). Within
      each level, number them in quartile order.
      Task tokens 0..15 go to targets (level order, then quartile order) and 16..31 to
      distractors.
    Log every rejected candidate count.

data.py  (public API)
  - Strings are encoded as bits uint32 (bit i = character i, a=0, b=1) plus a length
    uint8.
  - in_test_partition(bits, length): the first byte of SHA-256 of the ASCII string is
    below 32.
  - sample_strings(rng, n, length, partition): uniform strings of that length,
    rejection-sampled into the partition.
  - build_test_set(config): 241 strings per length 8..23 and 240 at length 24, drawn
    with replacement (task seed). It is shared by every task.
  - build_pool(config, task): 4096 train-partition strings per length (69,632 total),
    seeded from (task_seed, task_token).
  - assemble_mix(config, tasks, target_id, alpha, seed) returns per-example arrays
    (task_id, pool_index, batch_length) for T_max = 65536 examples:
      - In each octave interval [1,512], (512,1024], ..., (32768,65536], exactly
        round(alpha * interval_length) target slots, placed uniformly at random.
      - The other slots dealt to the 16 distractors in equal counts (+-1).
      - 1024 batches of 64, each of a single length, from a balanced length schedule
        (each length 60 or 61 batches) shuffled by the seed.
      - Each task's pool read in a seed-specific permutation without replacement
        within each length.
    It also returns per-checkpoint per-task counts for mixes.json. The data-order rng
    is seeded from (seed, target token, round(100*alpha)).
    Experiment 15 imports this function to rebuild every mix. It must be pure and
    deterministic.

model.py  (public API)
  - build_model(config, n_task_tokens=32): decoder-only, causal, pre-LayerNorm, GELU,
    learned positions (25). Vocabulary is a, b plus n_task_tokens. The input is
    [task token, c_1..c_l]. The output is a 2-way softmax at the last position. No
    weight tying.
    Assert 104,002 parameters at n_task_tokens = 32.
  - build_lstm(config, n_task_tokens=32): 84,290 parameters.
  - export_npz(model, path): float32 numpy arrays keyed by state_dict name, no
    pickle. load_npz(model, path).
  - evaluate(model, test_set, labels): fixed order (by length, then index), one batch
    per length, float32 forward, -log2 p(correct) accumulated in float64. Returns the
    mean bits, accuracy, per-length bits and per-string bits.
    run.py and verify.py both call this exact function.

run.py
  - train_cell(config, tasks, target_id, alpha, seed, out_dir, n_task_tokens=32,
    arch="transformer") is public (Experiment 15 imports it):
      torch.manual_seed(seed) before build_model, so all cells of a seed share an
      initialization.
      AdamW with linear warmup over 8 steps, then a constant rate.
      Evaluate and save a checkpoint at each T in T_checkpoints.
      Write the cell's metrics rows, its predictions npz and the checkpoints
      checkpoints/<seed>_<target_id>_<alpha:.2f>/T_<T>.npz atomically (temporary
      name, then rename) plus a done marker.
  - CLI:
      --check-only  Build tasks and data and run every invariant check (below). Print
                    one JSON receipt. No training.
      --pilot       Seed 11909, alpha = 1, all 16 targets, T to 131072 (2048 steps),
                    into results_pilot/.
      --output DIR  Default results/. Refuse if DIR exists and is not empty, unless
                    --resume.
      --resume      Skip cells with a done marker. Log the resume in manifest.json.
      --lstm        The optional exploratory LSTM arm after the main grid: alpha = 1,
                    8 seeds, 16 targets. Runs only if at least 45 minutes remain under
                    the cap. Saves only T = 65536 checkpoints.
      --only SEED:TARGET:ALPHA --output SCRATCH
                    Regenerate one cell for determinism checks. It never writes into
                    results/.
  - Run the 512 cells with ProcessPoolExecutor(max_workers=4), 1 torch thread each,
    longest-first.
  - Stop scheduling new cells when elapsed time passes max_wall_seconds. Record which
    cells are missing.
  - Write the shared output contract to results/:
      tasks.json: all fields in PROTOCOL Section 8.
      data/<task_id>_train.npz and data/<task_id>_test.npz (the shared test set, with
        that task's labels).
      mixes.json
      metrics.csv with columns seed, target_id, K, K_bits, alpha, T, step,
        excess_bits, accuracy, n_test, target_examples_seen,
        distractor_examples_seen, excess_bits_by_length (JSON list), wall_seconds.
      predictions/
      checkpoints/
      manifest.json: SHA-256 of every source and config.json, library versions, host,
        CPU model, start and end UTC, elapsed time, the check receipt, the pilot
        summary.
  - The --check-only receipt must include:
      the task table (id, n, K_bits, acceptance, sensitivity, quartile, tau);
      Hopcroft == Moore for all 32 tasks;
      pairwise language distinctness and non-complementarity;
      pool and test cardinalities per length;
      zero test strings in any pool;
      a schedule check for one cell at every alpha;
      the parameter count;
      a 20-step timing probe and the projected full-run hours.

laws.py
  - LAWS: M0_scale, M_align_eff, M_align_add, M_K_coef, M_K_exp, M_K_both, M_K_floor,
    plus the per-target power law. The K regressor is selectable: K_bits (primary) or
    n.
  - Censored residual on log2 epsilon at eps_res = 1e-3 (PROTOCOL Section 4.1).
  - scipy.optimize.least_squares (trf) with the bounds and the 64-start grid.
  - M_align_add is fit on epsilon in linear space and flags forecasts <= 0 as invalid.
  - predict(); t_star(params, K, alpha, eps_star); elasticities(params, K_center,
    eps_star) with the analytic e_alpha = eta and e_K from PROTOCOL Section 4.4.

independent_fit.py
  A different method: profile log A out in closed form, then L-BFGS-B over the
  exponents from the same starts. Objective and held-out forecasts must agree with
  laws.py to 1e-4 relative.

analyze.py
  Stage 1 reads only fit rows (filtered at read time). It fits every law per seed on
  the 252 fit points, picks the G2 K law by fit-cell RMSE, writes fits.json and
  forecasts_frozen.json, and prints that file's SHA-256 (paste it into CKPT).
  Stage 2 reads held-out rows and computes:
    - held-out RMSE for every law, overall and per slice (K = 16, alpha = 0.15,
      T = 65536);
    - T* table (tstar.csv) with censor flags;
    - M_K (seed-block and two-stage bootstrap);
    - E law-based in both units, E_mf with censor counts, and the E(eps*) curve with
      eps_flip;
    - the range ratio R;
    - eta and the M_align_eff versus M_align_add comparison;
    - the G5 gap ratio in bits and in log2 bits;
    - G6;
    - per-target gamma_j, Spearman(gamma_j, tau_j) and its partial version;
    - the K-versus-sensitivity regression with VIF;
    - the plateau census;
    - per-length loss.
  Bootstrap: 20,000 draws resampling seeds with replacement (analysis seed 29101126).
  The two-stage version also resamples the 4 targets within each level. Percentile
  95% intervals.
  Evaluate every gate G1-G7 exactly as PROTOCOL Section 5 states. Write analysis.json.

verify.py
  Never retrains. Checks, in order:
    1. source and config hashes against manifest.json;
    2. every task regenerated from the task seed equals tasks.json, and n is
       recomputed by both Hopcroft and Moore;
    3. every covariate recomputed (exact DP), plus a Monte Carlo cross-check;
    4. data/ and mixes.json regenerated byte-identical;
    5. every checkpoint reloaded into a fresh model via load_npz, evaluate() rerun, and
       every metrics.csv excess_bits and accuracy matched with
       max_abs_diff < 1e-9 bits;
    6. predictions matched;
    7. T* recomputed;
    8. both fitters rerun with agreement within 1e-4;
    9. every gate recomputed from scratch and compared to analysis.json.
  Writes results/verification.json with status PASS or FAIL and the counts.

report.py
  Writes experiments/11_dfa_exact_complexity_vs_alignment/RESULTS_dfa_complexity_alignment.md
  and a figure (dfa_complexity_alignment_results.png and .pdf) with four panels:
    (a) alpha = 1 learning curves by n (seed mean, with the band);
    (b) a T* heatmap over n x alpha at the primary eps*;
    (c) E versus eps* with the E = 1 line and eps_flip;
    (d) held-out forecasts versus observed for the G2 K law and M_align_eff.

test_dfa.py, test_data.py, test_model.py, test_laws.py (see UNIT TESTS)

.gitignore (experiment-local): results/checkpoints/, results_pilot/checkpoints/, *.tar,
*.tar.zst

UNIT TESTS (all must pass before --check-only; run each with the venv python)
test_dfa.py
  - Hopcroft versus Moore on >= 1,000 random DFAs with 2-10 states (rng seed 1): the
    same partition up to relabeling.
  - Every minimized DFA is equivalent to its source, both by product BFS and by brute
    force over all strings of length < n1 + n2.
  - Known minimal sizes:
      all strings -> 1;
      even number of a's -> 2;
      ends with "ab" -> 3;
      length mod 4 == 0 -> 4;
      even a's AND even b's -> 4.
  - canonical_form is invariant under random relabeling of non-start states.
  - sample_minimal_dfa returns exactly n minimal states for each n in {4,8,16,32}.
  - Exact DP acceptance, sensitivity and visitation versus Monte Carlo (20,000
    strings) within 4 standard errors on 20 random DFAs.
  - Complement detection works.
test_data.py
  - No test string appears in any pool (exact match on length and bits).
  - Per-length counts are 4096 in each pool, and 241/240 in the test set.
  - Labels equal a plain per-character Python DFA loop on 2,000 strings per task.
  - assemble_mix: exact target count per octave; distractor counts equal within +-1;
    length schedule 60/61; no pool index reused within a length in a run; identical
    output for the same seed and different output for another seed.
  - Token-weighted alpha (Varun's Eq. 2) equals example-weighted alpha within 0.01.
test_model.py
  - Parameter counts are 104,002 and 84,290.
  - An npz round trip gives identical logits (max abs diff 0).
  - Two 20-step train_cell runs with the same seed give bitwise-identical weights.
  - A different seed gives different weights.
  - evaluate() equals a naive per-string loop to 1e-12.
test_laws.py
  - On synthetic data on the real design grid (8 synthetic seeds, log-normal noise
    sd 0.05) generated from M_K_coef, M_K_exp and M_K_both with known parameters:
    recovery within 0.05 absolute for eta, gamma, gamma0, gamma1 and kappa.
  - The G3 logic returns G3a on coefficient-generated data and G3b on
    exponent-generated data.
  - The independent fitter agrees within 1e-4.
  - The censored residual is zero when both the forecast and the observation are
    below eps_res.
  - t_star is exact on a pure power law, and censor flags are correct.
  - Analytic e_K equals the finite difference within 1e-6.
  - M_align_add negative forecasts are flagged invalid.
  - A bootstrap over identical seeds collapses to the point estimate.

ORDER OF WORK
 1. Branch, read, set up the environment.
 2. Write the code and tests. Run every test file. Commit the code ("Experiment 11:
    code and tests").
 3. run.py --check-only. Save the receipt to results_check.json. Commit it.
 4. Pilot (disclosed). run.py --pilot. Apply PROTOCOL Section 2.6 rules P1-P3
    mechanically. Write a "Pilot disclosure" paragraph into PROTOCOL Section 2.6 with
    the numbers: per-target T*(0.1) at alpha = 1, step time, projected hours, any
    window or eps* change, and how many targets improve from 2^16 to 2^17.
    Seed 11909 is never reused, and pilot outputs are never pooled.
 5. Freeze. In PROTOCOL.md, replace the Status line with "Frozen on MM-DD-YYYY before
    any gated Experiment 11 training." Apply only pilot-rule changes. Commit
    PROTOCOL.md (and config.json if P1 or P2 fired) alone as
    "Freeze Experiment 11 protocol". Put that commit hash in CKPT. From now on,
    Sections 2-5 of PROTOCOL.md and every gate value in config.json are read-only.
 6. Full run, detached:
      nohup ~/venvs/exp11/bin/python expt_v1/run.py --output expt_v1/results > run.log 2>&1 &
    or a named tmux session. Log the start and end in CKPT. Optional: --lstm afterwards.
 7. verify.py must print PASS. On FAIL, do not edit outputs. Diagnose, fix only
    verifier bugs, log each step in CKPT, and rerun verify.
 8. analyze.py (stage 1, paste the hash into CKPT, then stage 2), then report.py.
 9. Write CKPT_dfa_complexity_alignment.md, a dated log (MM-DD-YYYY) of every step:
    commands, hashes, wall times, deviations, failures.
    Pack the checkpoints:
      tar -cf results/checkpoints_dfa.tar -C results checkpoints
    Write ARTIFACT_SHA256.json covering every results file and the tar.
10. Archive: git mv experiments/11_dfa_exact_complexity_vs_alignment
    experiments/archive/11_dfa_exact_complexity_vs_alignment. Add a row to
    experiments/README.md and to experiments/archive/README.md. Change the Experiment
    11 link in experiments/ideas/complexity_vs_alignment/README.md to the archive
    path. The checkpoints and the tar stay on disk, uncommitted (Experiments 14 and 15
    run on this box and read them in place).
11. git push -u origin exp11-dfa-complexity-alignment, then open a pull request with
    gh pr create:
      title "Experiment 11: exact DFA complexity versus alignment";
      body: Summary (at most 10 bullets, every gate with its verdict) and
      Test plan (2-3 bullets: unit tests, verify.py PASS, check-only receipt).
    Do not merge. Merging waits on Brando.

RESULTS FILE MUST CONTAIN
  - A gate table: gate, statistic, estimate, 95% interval(s), threshold, verdict, for
    G1-G7. Same prominence for every verdict.
  - The target table: id, n, K_bits, acceptance, sensitivity, tau, T* per eps*.
  - Every law's held-out RMSE by slice.
  - eta, with its reading: transfer, pure effective data, or interference.
  - E at 0.3/0.1/0.03 bits and eps_flip.
  - The range ratio R.
  - The K-versus-sensitivity regression.
  - The plateau census and the LSTM arm (if run).
  - Pilot disclosure, deviations, limitations (PROTOCOL Section 9), and the
    artifact location.
  - Wording: "K is exact for the DFA description language". Never "Kolmogorov
    complexity measured".

HARD RULES
  - Never change a gate, threshold, split, seed list or eps* after the freeze commit.
  - Never drop a seed, task, checkpoint or forecast. Censored and plateaued runs stay
    in, with flags.
  - No hidden reruns. Every run, resume, crash and retry is logged in CKPT.
  - If a unit test or check fails twice for the same cause after a fix attempt, stop.
    Write the blocker in CKPT, push the branch, and open a draft pull request that
    describes it.
  - Bounded resources: 4 workers, 1 thread each, RAM under 12 GB, artifacts under
    2 GB, wall cap 18,000 s.
  - No model-provider API calls. No credentials, tokens or keys in the repo or logs.
  - No AI or tool attribution in commits, files or the pull request.
  - Plain English. Expand acronyms on first use. Dates MM-DD-YYYY. End every new
    markdown file with a "**TLDR-end:** [tdc: ...]" line.

DELIVERABLE CHECKLIST
  [ ] expt_v1/: config.json, dfa.py, data.py, model.py, run.py, laws.py,
      independent_fit.py, analyze.py, verify.py, report.py, test_dfa.py,
      test_data.py, test_model.py, test_laws.py, requirements.txt, .gitignore
  [ ] all four test files pass; check-only receipt committed
  [ ] pilot disclosed in PROTOCOL Section 2.6 and CKPT; seed 11909 not reused
  [ ] freeze commit of PROTOCOL.md before any gated run; hash in CKPT
  [ ] 512/512 cells complete (or the wall-cap shortfall logged), within 18,000 s
  [ ] results/: tasks.json, data/, mixes.json, metrics.csv (4,096 rows),
      predictions/, checkpoints/ (on disk), tstar.csv, fits.json,
      forecasts_frozen.json, analysis.json, manifest.json, verification.json (PASS)
  [ ] RESULTS_dfa_complexity_alignment.md with all gates G1-G7 and the figure (png, pdf)
  [ ] CKPT_dfa_complexity_alignment.md dated log
  [ ] ARTIFACT_SHA256.json (including checkpoints_dfa.tar)
  [ ] folder moved to experiments/archive/; rows in experiments/README.md and
      archive/README.md; overview link updated
  [ ] branch exp11-dfa-complexity-alignment pushed; pull request open, not merged
```

**TLDR-end:** [tdc: Expt 11 prompt] A self-contained coding-agent prompt that builds the exact-DFA task generator, mix assembly, a 104,002-parameter transformer and the law fitters, tests them against brute force and synthetic data, runs a disclosed pilot, freezes the protocol, trains 512 CPU runs under a 5-hour cap, replays every checkpoint, and reports gates G1–G7 in a pull request.
