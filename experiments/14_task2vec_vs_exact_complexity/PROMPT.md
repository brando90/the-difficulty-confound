# Coding-agent prompt: run Experiment 14 end to end

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/14_task2vec_vs_exact_complexity/PROMPT.md>

Paste the block below into Claude Code or Codex at the repository root. It is self-contained.

```text
ROLE AND GOAL
You are a careful research engineer. Run Experiment 14 of the repository
https://github.com/brando90/the-difficulty-confound end to end. The experiment asks
whether the Task2Vec Fisher-embedding norm tracks exact task complexity K and the
data a task needs, and whether Brando's diversity coefficient measures coverage
rather than complexity. Build, test, freeze, run, verify, report, archive, and
open a pull request. Do not merge it; merging waits on Brando.

BRANCH
git checkout main && git pull && git checkout -b exp14-task2vec-complexity

READ FIRST, IN THIS ORDER
1. experiments/14_task2vec_vs_exact_complexity/PROTOCOL.md: the design. It is the
   contract. Sections 2-5 must not change after your freeze commit.
2. experiments/ideas/complexity_vs_alignment/README.md: the program, notation (epsilon
   excess loss in bits, T examples seen, alpha target share, K exact complexity),
   and the shared rules.
3. experiments/ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md: what Task2Vec
   does and does not say about Kolmogorov complexity.
4. Experiment 11 and 12 folders: experiments/11_dfa_exact_complexity_vs_alignment/
   and experiments/12_dsl_shortest_program_complexity_vs_alignment/ (or their copies
   under experiments/archive/). Read their PROTOCOL.md, config.json, RESULTS file,
   and the code that defines the model, vocabulary, input sampler, task generator,
   exact-K computation, the laws, and how a training mix is assembled.
5. experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/: the code
   layout and verification style to mirror (run.py, verify.py, analyze.py,
   independent_fit.py, test_laws.py).
6. The Task2Vec reference code, read-only:
   https://github.com/awslabs/aws-cv-task2vec (task2vec.py: montecarlo_fisher,
   _fit_classifier, _get_loader, extract_embedding; task_similarity.py: cosine)
   and the diversity-coefficient code https://github.com/brando90/beyond-scale-div-coeff
   (src/diversity/div_coeff.py: get_diversity_coefficient; src/diversity/task2vec.py:
   _finetune_classifier, montecarlo_fisher_autoregressive). Copy no code with a
   license problem; reimplement the same arithmetic and cite the function you mirror
   in a comment.

PRECONDITION (STOP IF FALSE)
Experiments 11 and 12 must be complete: each expt_v1/results/ holds tasks.json,
data/<task_id>_{train,test}.npz, mixes.json, metrics.csv, checkpoints/,
manifest.json and ARTIFACT_SHA256.json, and its verify.py printed PASS. If not,
write the blocker in CKPT_task2vec_complexity.md and stop.

ENVIRONMENT (CPU box: 4 cores, 15 GB RAM, no GPU, no torch preinstalled)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==<the exact version in Experiment 11's requirements.txt> \
    --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy   (use Experiment 11's pinned versions where it pins them)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Pin every installed version in expt_v1/requirements.txt. In code, call
torch.set_num_threads(1); run at most 4 worker processes; keep RAM under 12 GB.

FILES TO CREATE under experiments/14_task2vec_vs_exact_complexity/expt_v1/
- config.json: every constant in the PROTOCOL.
  - Seeds: probe seeds [14101, 14202, 14303, 14404]; random-init probe seed 14505;
    pilot seed 14909 (pilot only, never reused); task_sampling_seed 20261014;
    analysis_seed 29101426; bootstrap_draws 20000.
  - Counts: examples_per_task 2048; probe_pretrain_examples 65536;
    task_dropout 0.5; diversity_batches 12; diversity_batch_size 512;
    coverage_m [1,2,4,8,16]; coverage_replicates 2; noise_rates [0.10, 0.25];
    noise_tasks_per_level 4.
  - Estimator settings for E0 to E3 exactly as in the PROTOCOL table.
  - Gate thresholds exactly as in PROTOCOL Section 5.
  - workers 4, threads 1, max_wall_seconds 14400.
  - Paths to the Experiment 11 and 12 results.
- tasks14.py: loads Experiment 11/12 tasks.json and data, and imports (does not
  copy) their generator, input sampler and exact-K functions via sys.path. Before
  use, it asserts that the SHA-256 of each imported source equals the value in that
  experiment's manifest.json. It draws or loads the probe-pretraining tasks
  (16 per K level), the fresh tasks (16 per K level) and the label-noise variants.
  It enforces pairwise disjointness: canonical minimal DFAs (deterministic finite automata) for language
  equality; full truth tables for DSL (domain-specific language) functions. It writes results/tasks14.json and fixed
  2,048-example sets per task.
- probe.py: builds the probe from the imported Experiment 11/12 model class.
  - Unties the input embedding from the output layer if they are tied (the only
    allowed change).
  - Adds one reserved null task token.
  - Pretrains with task dropout 0.5 on 65,536 fresh examples, using Experiment
    11/12's optimizer, schedule and batch size.
  - Saves float32 weights to results/probes/<family>_<seed>.npz with no pickle.
  - Builds the function-preserving rescaled copy (query x2 / key /2, value x2 /
    attention-output /2, weights and biases) and asserts max output difference
    < 1e-5.
- task2vec.py: estimators E0 (reference Monte Carlo), E1 (empirical labels),
  E2 (per-example Monte Carlo, 5 label draws per example) and E3 (diversity-paper
  pipeline), exactly as in the PROTOCOL.
  - Head = the probe's own output head (Experiment 11: 2-way accept/reject head at
    the last position; Experiment 12: its output head over the 10 output bits),
    initialized from the probe's trained head weights.
  - Class-balanced sampler as in the reference _get_loader.
  - Head fitting on cached hidden states is allowed (the body is frozen, no dropout).
  - Aggregation as in the reference extract_embedding: every non-head module with a
    .weight, grad2.reshape(rows, -1).mean(1), concatenated in named_modules order,
    biases ignored.
  - Also: symmetric cosine distance (the reference task_similarity.cosine with
    1e-8), the diversity coefficient (interleaved and single-task batches), the
    cross-diversity coefficient, Rao's quadratic entropy, and the lower and upper
    reference-bound datasets.
  - Every random draw comes from a torch.Generator seeded from (seed, family, task
    or mix id, probe, estimator), so any embedding can be recomputed exactly.
  - Experiment 15 imports this module: keep its public functions documented and
    stable.
- run.py: stages probes, embed and diversity (default: all).
  - --check-only: build tasks, check counts against the PROTOCOL table, check
    disjointness, recompute K for every new task, check that the null token never
    co-occurs with a task token in embedding inputs, check that the total-variation
    distance between task input distributions is 0, and print expected work counts.
    Trains nothing.
  - --pilot: one DFA probe with seed 14909, 8 tasks x all estimators, 2 mixes of
    diversity. Writes timings to results_pilot/. The pilot is disclosed and its
    seed is never reused.
  - --output DIR: refuses to overwrite an existing non-empty DIR.
  - Writes results/manifest.json (source SHA-256s, Experiment 11/12 SHA-256s,
    versions, hostname, wall time) and stops cleanly at max_wall_seconds, leaving a
    resumable state and a log line.
- stats.py (takes the place of laws.py; this experiment fits no scaling law):
  - Spearman with average ranks, and partial Spearman by rank residualization.
  - T*(epsilon*) from Experiment 11/12 metrics.csv at alpha = 1: log-log
    interpolation, right-censored = +inf, median over seeds; epsilon* is read from
    Experiment 11/12's frozen config or protocol and never chosen here.
  - Kendall's W; the stratified bootstrap over tasks, probes, seed blocks and batch
    pairs.
- independent_stats.py (takes the place of independent_fit.py): a second
  implementation of every statistic in plain numpy with its own rank code, no
  scipy. It must agree with stats.py to 1e-10 on every reported number.
- analyze.py: computes Gates 1-5 and every reported (non-gated) analysis in
  PROTOCOL Section 4. Writes results/gates.json, task_table.csv, diversity.csv and
  bootstrap.npz.
- verify.py: everything in PROTOCOL Section 8; prints PASS or the first failure.
- report.py: writes RESULTS_task2vec_complexity.md and the figure
  task2vec_vs_K.png.
  - Figure panels: norm versus K per family with probe spread; partial plot after
    covariates; norm versus T*; diversity versus H on the coverage mixes.
  - Every gate appears in one table with estimate, 95% interval, threshold and
    verdict.
- test_task2vec.py, test_stats.py, test_tasks14.py: the unit tests below.
- requirements.txt: exact pins.

UNIT TESTS THAT MUST PASS BEFORE ANY RUN
1. Fisher correctness: use a logistic-regression model with fixed features on a
   label-balanced dataset (so the class-balanced sampler is uniform). The analytic
   diagonal FIM (Fisher information matrix) is mean_x p(1-p) x_i^2.
   - E2 with 200 draws per example matches it within 2%.
   - E0 x batch_size matches it within 5%.
   - E1 equals a hand computation of the squared batch-mean gradients (1e-6).
   - Its per-example form equals mean_x (y-p)^2 x_i^2 (1e-6).
2. Aggregation: on a hand-built module set, the embedding equals the reference
   arithmetic (per-row mean of grad2) in the reference order; the head is excluded.
3. Distance: symmetric cosine equals a literal transcription of the reference
   cosine on 100 random positive vectors (1e-12). The distance of an embedding to
   itself is 0.
4. The rescaled probe is function-preserving (max output difference < 1e-5) and
   changes the E0 embedding (it should).
5. Head fitting on cached features equals head fitting through the full frozen
   model (max parameter difference 1e-6).
6. Determinism: the same seeds reproduce an embedding bit for bit on 1 thread.
7. Tasks: role counts equal the PROTOCOL table. No two tasks across roles share a
   language or truth table. Recomputed K equals recorded K. Each label-noise
   variant flips exactly round(rate * 2048) labels.
8. Statistics: Spearman and partial Spearman match scipy and independent_stats.py.
   The bootstrap is reproducible from the analysis seed. T* interpolation is exact
   on a synthetic power law. A censored curve gets +inf.
If a test fails twice after honest fixes, write the blocker in CKPT and stop.

ORDER OF WORK
1. Build the files. Run the unit tests. Run run.py --check-only.
2. Run the disclosed pilot (seed 14909). Record its timings. Apply the PROTOCOL's
   pre-declared fallback (12 -> 8 diversity batches) only if the projected total
   exceeds 2.5 h. Write the pilot and that decision into PROTOCOL.md (a short
   "Pilot disclosure" paragraph at the end of Section 2).
3. FREEZE: commit PROTOCOL.md (plus config.json and the code) alone, with the
   message "Freeze Experiment 14 protocol". No gated embedding may exist before
   this commit.
4. Full run under the wall cap: run.py --output expt_v1/results.
5. verify.py must print PASS. If it fails, fix the cause, log it in CKPT, and rerun
   only what the fix touches; every rerun is logged.
6. analyze.py, then report.py. RESULTS_task2vec_complexity.md must contain:
   - every gate (1-5) with equal prominence, SUPPORTED or NOT_ESTABLISHED;
   - probe-seed sensitivity (Kendall's W, pairwise Spearman);
   - parameterization sensitivity (rescaled probe, E0-E3, L1, L2 and log-sum,
     layer subsets, random-init probe);
   - the mechanism analysis (norm versus head error versus K);
   - the incremental value over exact K for T*;
   - the reference bounds;
   - the Section 6 row that the outcome matches;
   - whether the real-data handoff criteria in PROTOCOL Section 9 are met.
7. CKPT_task2vec_complexity.md: a dated log (MM-DD-YYYY) of every step, pilot,
   failure, rerun and decision.
8. ARTIFACT_SHA256.json over every file in results/ and the reports.
9. Archive: git mv experiments/14_task2vec_vs_exact_complexity
   experiments/archive/14_task2vec_vs_exact_complexity. Fix relative links that
   the move breaks. Add one row to experiments/README.md (setup and status, one
   line, numbers from RESULTS).
10. Push the branch and open a pull request with gh pr create. Summary: at most 10
    bullets, every gate verdict with its number. Test plan: 2-3 bullets (unit
    tests, verify.py PASS, check-only). Merging waits on Brando.

HARD RULES
- Never change a gate, threshold, estimator, seed or count after the freeze commit.
- Never drop a task, probe, mix, seed or outcome. Censored T* stays in as +inf.
- No hidden reruns: every run, including failed ones, is in CKPT.
- If a unit test fails twice, stop and write the blocker in CKPT.
- Bounded resources: 4 workers x 1 thread, RAM < 12 GB, max_wall_seconds 14400,
  artifacts < 2 GB.
- Never modify Experiment 11/12 files. Import them read-only and check their hashes.
- No calls to any model-provider API. No credentials in the repository.
- No tool or AI attribution in commits, docs or the pull request.
- Plain English; expand acronyms on first use; dates MM-DD-YYYY.

DELIVERABLE CHECKLIST
[ ] expt_v1/ code, config.json, requirements.txt, tests, all passing
[ ] run.py --check-only output saved in CKPT
[ ] pilot disclosed in PROTOCOL.md; freeze commit made before any gated embedding
[ ] results/ complete: tasks14.json, probes/, embeddings.npz, batch_embeddings/,
    diversity.csv, task_table.csv, gates.json, bootstrap.npz, manifest.json
[ ] verify.py prints PASS
[ ] RESULTS_task2vec_complexity.md with Gates 1-5 at equal prominence, the
    sensitivity analyses, and the handoff verdict
[ ] task2vec_vs_K.png
[ ] CKPT_task2vec_complexity.md (dated log)
[ ] ARTIFACT_SHA256.json
[ ] folder moved to experiments/archive/14_task2vec_vs_exact_complexity/, links
    fixed, row added to experiments/README.md
[ ] branch exp14-task2vec-complexity pushed, pull request open, not merged
```

**TLDR-end:** [tdc: Expt 14 prompt] A self-contained coding-agent prompt that builds Task2Vec estimators matching the reference code, pretrains frozen probes, embeds every Experiment 11/12 task, then freezes, runs, verifies and reports the five numeric gates on whether the Task2Vec norm tracks exact complexity and data cost and whether the diversity coefficient tracks coverage.
