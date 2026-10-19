# Coding-agent prompt: run Experiment 15 end to end

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/15_alignment_alpha_recovery/PROMPT.md>

Paste the block below into Claude Code or Codex at the repository root. It is self-contained.

```text
ROLE AND GOAL
You are a careful research engineer. Run Experiment 15 of the repository
https://github.com/brando90/the-difficulty-confound end to end. The experiment asks
whether the measured alignment alpha-hat of the alignment-scaling paper with Varun
(token-weighted per-example scores, its Eq. 2) recovers the true target share alpha
that Experiments 11 and 12 set by hand. It asks whether alpha-hat stays flat across
exact target complexity K at fixed alpha, and whether a calibrated alpha-hat can
replace alpha in the best scaling law. A new graded-relevance training arm tests
whether chance-corrected relevance predicts loss better than target share. Build,
test, freeze, run, verify, report, archive, and open a pull request. Do not merge
it; merging waits on Brando.

BRANCH
git checkout main && git pull && git checkout -b exp15-alpha-recovery

READ FIRST, IN THIS ORDER
1. experiments/15_alignment_alpha_recovery/PROTOCOL.md: the design. It is the
   contract. Sections 2-5 must not change after your freeze commit.
2. experiments/ideas/complexity_vs_alignment/README.md: the program, notation (epsilon
   excess loss in bits, T examples seen, alpha target share, K exact complexity),
   the law family (M0_scale, M_align_eff, M_align_add, M_K_coef, M_K_exp,
   M_K_both), fit and held-out cells, and the shared rules.
3. experiments/ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md, Section 4(c):
   why Task2Vec distance is at best a monotone, uncalibrated transform of alpha.
4. Experiment 11 and 12 folders: experiments/11_dfa_exact_complexity_vs_alignment/
   and experiments/12_dsl_shortest_program_complexity_vs_alignment/ (or their copies
   under experiments/archive/). Read PROTOCOL.md, config.json, the RESULTS file (which
   law is best on held-out cells), and the code for the model, vocabulary, sequence
   format, input sampler, task generator, exact-K and minimization, mix assembly,
   training loop, laws and fitter.
5. Experiment 14 folder (experiments/14_task2vec_vs_exact_complexity/ or its archive
   copy): probe.py, task2vec.py, results/probes/, manifest.json.
6. experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/: the code
   layout and verification style to mirror.
7. Outside references, read-only, cite them in comments where you mirror their
   arithmetic:
   - ZIP-FIT, arXiv 2410.18194 (gzip normalized compression distance, Algorithm 1).
   - LESS, arXiv 2402.04333, code https://github.com/princeton-nlp/LESS. Ours is a
     simplified frozen-gradient cosine, not LESS itself.
   - The alignment-scaling paper's code https://github.com/madanva/alignment-scaling-laws
     (private). If you can read it, take the exact compression-ratio similarity
     formula for measure (a) from it and record the file and commit. If not, use the
     PROTOCOL's literal reading and say so in RESULTS. Cite that work only as "the
     alignment-scaling paper with Varun" with that link; no other link, no author
     list.

PRECONDITION (STOP IF FALSE)
Experiments 11, 12 and 14 must be complete, each with verify.py PASS. Experiments
11/12 must provide results/tasks.json, data/, mixes.json, metrics.csv,
checkpoints/, manifest.json and ARTIFACT_SHA256.json. Experiment 14 must provide
results/probes/<family>_{14101,14202,14303,14404}.npz and task2vec.py. If any is
missing, write the blocker in CKPT_alpha_recovery.md and stop. (The graded-arm
training needs only Experiment 11. If Experiment 12 or 14 is late, you may build and test that part,
but nothing gated runs before the freeze, and the freeze needs the whole pilot.)

ENVIRONMENT (CPU box: 4 cores, 15 GB RAM, no GPU, no torch preinstalled)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==<the exact version in Experiment 11's requirements.txt> \
    --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy zstandard scikit-learn   (Experiment 11's pins where it pins)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Pin every installed version in expt_v1/requirements.txt. In code, call
torch.set_num_threads(1); run at most 4 worker processes; keep RAM under 12 GB.

FILES TO CREATE under experiments/15_alignment_alpha_recovery/expt_v1/
- config.json: every constant in the PROTOCOL.
  - Seeds: graded training seeds [15101, 15202, 15303, 15404, 15505, 15606, 15707,
    15808]; pilot seed 15909 (pilot only, never reused); task_sampling_seed
    20261015; analysis_seed 29101526; bootstrap_draws 20000.
  - Probes: [14101, 14202, 14303, 14404].
  - Counts: references_per_target 2048; scored_units_per_task 4096; ncd_references
    128; ncd_units_per_task 512; task2vec_batches 8; task2vec_batch_size 256.
  - Dictionary-size rule; graded rewiring levels [1, 2, 4, "all"] with 2
    distractors per level; target acceptance band [0.40, 0.60]; distractor band
    [0.30, 0.70]; mixes m1-m5 exactly as in the PROTOCOL table.
  - Gate thresholds exactly as in PROTOCOL Section 5.
  - workers 4, threads 1, max_wall_seconds 14400, graded_fallback_minutes 90.
  - Paths to Experiments 11, 12 and 14.
- mixes15.py: imports (does not copy) Experiment 11/12's mix-assembly function and
  rebuilds each seed's exact training multiset. Asserts per-task counts equal
  mixes.json and SHA-256 of imported sources equal their manifests. Selects the
  references (unused target training examples first, then fresh draws with seed
  20261015, logged) and the 4,096 scored units per task. Asserts that references
  and scored units are disjoint as index sets. Serializes examples without the task
  token through one fixed token-to-byte table (saved to results/).
- scores.py: measures a, a-prime, a2, b1, b2, c, c2, d, e, e2, the oracle and the
  random score, plus the controls (task token kept, permuted-label references,
  input-only). Min-max over each target's pool P_T. The stratified Eq. 2 estimator.
  - b1 and d load Experiment 14 probes with the null task token.
  - e and e2 call Experiment 14's task2vec.py E3 pipeline (imported, hash-checked).
  - Only scores are stored, never per-example gradients.
- graded.py: draws the 4 graded targets and 32 rewired distractors.
  Re-minimizes with Experiment 11's Hopcroft code and enforces language
  disjointness. Computes agreement exactly (product-automaton dynamic programming)
  when Experiment 11's sampler is uniform given length, otherwise by 2^20 Monte
  Carlo draws. Computes kappa relevance, alpha_soft, alpha_share and alpha_agree.
  Writes data, mixes and tasks in the shared contract. Trains with Experiment 11's
  imported training function, unchanged except for the task-token table size.
- run.py: stages refs, scores, graded (default: all).
  - --check-only: rebuild all mixes and assert counts; check reference/unit
    disjointness; check that no stripped serialization contains a task token;
    print input total-variation distances, reference coverage and graded
    agreement table. Enforce the graded construction check (mean kappa falls with
    r, at least 0.60 at r = 1, at most 0.20 at r = all). Print work counts.
    Trains nothing and scores nothing gated.
  - --pilot: seed 15909. Score 2 DFA (deterministic finite automaton) and 2 DSL
    (domain-specific language) targets with every measure on probe
    14101; train 5 graded runs. Writes timings to results_pilot/ and projects the
    full wall time. Disclosed; the seed is never reused.
  - --output DIR: refuses to overwrite an existing non-empty DIR.
  - Writes results/manifest.json (source SHA-256s, upstream SHA-256s, versions,
    hostname, wall time). Stops cleanly at max_wall_seconds, leaving a resumable
    state and a log line.
- laws.py: imports Experiment 11/12's law forms and fitter (hash-checked). Adds:
  isotonic calibration (increasing, linear interpolation, clip at ends) fit only on
  pairs outside the held-out K and alpha levels; the Gate 2 fixed-effects slope
  regression; leave-one-mix-out for Gate 4a; Spearman with average ranks; the
  stratified bootstrap of PROTOCOL Section 4.
- independent_fit.py: a second implementation of every law fit, the isotonic
  calibrator and the slope regression in plain numpy (no scipy, no scikit-learn).
  Must agree with laws.py to 1e-4 relative on every law number and 1e-10 on every
  statistic.
- analyze.py: Gates 1, 2, 3, 4a, 4b and every reported analysis in PROTOCOL
  Section 4, for every measure. Writes results/gates.json, alpha_hat.csv,
  decomposition.csv, controls.csv, laws.json and bootstrap.npz.
- verify.py: everything in PROTOCOL Section 8; prints PASS or the first failure.
- report.py: writes RESULTS_alpha_recovery.md and alpha_recovery.png.
  - Figure panels: alpha-hat versus alpha per K level (measure c, with the oracle);
    the Gate 2 slope per measure with its band; true-alpha versus calibrated versus
    raw law RMSE (root-mean-square error); graded arm loss versus alpha_share and alpha_soft.
  - Every gate in one table with estimate, 95% interval, threshold and verdict.
- test_scores.py, test_graded.py, test_laws.py: the unit tests below.
- requirements.txt: exact pins.

UNIT TESTS THAT MUST PASS BEFORE ANY RUN
1. Eq. 2: on a hand-built pool, alpha-hat equals the formula by hand (1e-12). With
   equal lengths it equals m_D + (m_T - m_D) * alpha. The stratified estimator
   equals the exact Eq. 2 when the scored units are the whole task file.
2. Min-max: scores lie in [0, 1] with 0 and 1 attained; a constant score raises
   a clear error instead of dividing by zero.
3. Compression: NCD (normalized compression distance) of a string with itself is near 0, and NCD of unrelated random
   bytes is near 1 (gzip, 1 KB strings). The zstd dictionary is trained only on
   references: a test passes a pool example into the trainer and expects an
   assertion error.
4. Stripping: a serialized example never contains a task-token byte; the task
   token-kept control equals the stripped one plus exactly one task-token byte.
5. Frozen gradient: the per-example gradient equals a finite-difference check on a
   tiny model (1e-4 relative). The embedding tables are excluded. The cosine with
   itself is 1.
6. Task2Vec fixed head (e2): on a tiny model with a fixed head, the exact diagonal
   Fisher (expectation over model labels, per example) of a 50/50 batch equals the
   mean of the two single-task Fishers (1e-6).
7. Graded: rewiring r transitions changes exactly r table entries before
   minimization. Exact agreement equals 2^16 Monte Carlo draws within 0.01 on 3
   pairs. Kappa equals 2 * agree - 1 when p_T = 0.5. Re-minimized K equals a
   brute-force minimization on 5-state examples.
8. Calibration and leakage: the isotonic calibrator is monotone; the fit never
   reads a held-out row (assert on index sets); the slope regression recovers a
   planted slope of -0.05 within 0.005 on synthetic data.
9. Laws: laws.py and independent_fit.py agree on a synthetic grid; the
   leave-one-mix-out never trains on the left-out mix.
10. Determinism: the same seeds reproduce scores bit for bit on 1 thread.
If a test fails twice after honest fixes, write the blocker in CKPT and stop.

ORDER OF WORK
1. Build the files. Run the unit tests. Run run.py --check-only and save its output
   in CKPT.
2. Run the disclosed pilot (seed 15909). Record timings. Apply the PROTOCOL's
   pre-declared fallback (8 -> 4 graded seeds, 15101-15404) only if the graded arm
   projects to more than 90 minutes on 4 workers. Write the pilot and that decision
   into PROTOCOL.md (a short "Pilot disclosure" paragraph at the end of Section 2),
   including whether measure (a) uses the paper's code formula or the literal
   reading.
3. FREEZE: commit PROTOCOL.md (plus config.json and the code) alone, with the
   message "Freeze Experiment 15 protocol". No gated score and no graded run may
   exist before this commit.
4. Full run under the wall cap: run.py --output expt_v1/results.
5. verify.py must print PASS. If it fails, fix the cause, log it in CKPT, and rerun
   only what the fix touches; every rerun is logged.
6. analyze.py, then report.py. RESULTS_alpha_recovery.md must contain:
   - Gates 1, 2, 3, 4a and 4b with equal prominence, SUPPORTED or NOT_ESTABLISHED;
   - the same statistics, without verdicts, for every other measure and the oracle;
   - the affine decomposition (m_D, m_T - m_D) versus K;
   - the controls (leak, permuted labels, input-only, random);
   - raw versus calibrated alpha-hat in the law;
   - the Section 6 row that the outcome matches;
   - if Gate 2 fails, the stated consequence for the alignment-scaling law;
   - the transfer table of PROTOCOL Section 9.
7. CKPT_alpha_recovery.md: a dated log (MM-DD-YYYY) of every step, pilot, failure,
   rerun and decision.
8. ARTIFACT_SHA256.json over every file in results/ and the reports.
9. Archive: git mv experiments/15_alignment_alpha_recovery
   experiments/archive/15_alignment_alpha_recovery. Fix relative links that the
   move breaks. Add one row to experiments/README.md (setup and status, one line,
   numbers from RESULTS).
10. Push the branch and open a pull request with gh pr create. Summary: at most 10
    bullets, every gate verdict with its number. Test plan: 2-3 bullets (unit
    tests, verify.py PASS, check-only). Merging waits on Brando.

HARD RULES
- Never change a gate, threshold, measure, seed or count after the freeze commit.
- Never drop a target, mix, measure, probe, seed or outcome; nonfinite forecasts
  count as invalid, not as missing.
- Never fit anything on held-out rows, and never let pool examples into the
  references or the zstd dictionary.
- No hidden reruns: every run, including failed ones, is in CKPT.
- If a unit test fails twice, stop and write the blocker in CKPT.
- Bounded resources: 4 workers x 1 thread, RAM < 12 GB, max_wall_seconds 14400,
  artifacts < 2 GB.
- Never modify Experiment 11, 12 or 14 files. Import them read-only and check
  their hashes.
- No calls to any model-provider API (Sentence-BERT is not used; measure b1 uses
  the Experiment 14 probes). No credentials in the repository.
- No tool or AI attribution in commits, docs or the pull request.
- Plain English; expand acronyms on first use; dates MM-DD-YYYY.

DELIVERABLE CHECKLIST
[ ] expt_v1/ code, config.json, requirements.txt, tests, all passing
[ ] run.py --check-only output saved in CKPT
[ ] pilot disclosed in PROTOCOL.md; freeze commit made before any gated score or
    graded run
[ ] results/ complete: references.npz, scored_units.npz, scores/, alpha_hat.csv,
    decomposition.csv, controls.csv, graded/ (shared contract), laws.json,
    gates.json, bootstrap.npz, manifest.json
[ ] verify.py prints PASS
[ ] RESULTS_alpha_recovery.md with Gates 1-4b at equal prominence, all measures,
    controls, raw versus calibrated, and the transfer table
[ ] alpha_recovery.png
[ ] CKPT_alpha_recovery.md (dated log)
[ ] ARTIFACT_SHA256.json
[ ] folder moved to experiments/archive/15_alignment_alpha_recovery/, links fixed,
    row added to experiments/README.md
[ ] branch exp15-alpha-recovery pushed, pull request open, not merged
```

**TLDR-end:** [tdc: Expt 15 prompt] A self-contained coding-agent prompt that rebuilds every Experiment 11/12 mix, scores it with the alignment-scaling paper's Eq. 2 under stripped-token compression, embedding, gradient and Task2Vec measures, trains a 160-run graded-relevance arm, then freezes, runs, verifies and reports the numeric gates on recovery, complexity-neutrality, downstream law use and graded relevance.
