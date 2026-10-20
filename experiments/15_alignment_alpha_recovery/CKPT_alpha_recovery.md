# CKPT: Experiment 15, can measured alignment recover the true alignment?

Dated log (MM-DD-YYYY, times US Pacific). Box: 4 cores, 15 GB, no GPU, torch 2.14.1+cpu, Python venv ~/.venvs/tdc. Branch results/cpu-expts-11-15. Experiments 11, 12 and 14 are read, never written: their sources, results and probes are hash-checked against `expt_v1/upstream_sha256.json` on import.

## 10-19-2026

- 16:00 Started. PROTOCOL.md and PROMPT.md identical to the design snapshot (commit 45302443) and to the design author's live tree; origin/main (20d8f2a0) has no change to this folder.
- 16:00-16:30 Code: common.py (upstream imports with SHA-256 checks, serialization), mixes15.py (mix reconstruction, references, scored units), scores.py (every measure), jobs.py (scoring jobs per arm), graded.py (graded-relevance arm), laws.py, independent_fit.py, run.py, analyze.py, verify.py, report.py, three test files, requirements.txt (exact pins of the venv).
- Smoke tests of the scoring code on scratch copies (not results): per-example gradients about 195 per second (DFA) and 530 per second (DSL) on an idle core; Task2Vec 72 s (DFA) and 22 s (DSL) per (target, probe).
- Unit tests, first run: 2 failures, both in test code. `test_ncd_self_and_random`: gzip NCD(x, x) on a 1 KB random string is 0.058 (header and back-reference cost), so the bound was loosened to < 0.1 with a comment. `test_frozen_gradient_finite_difference`: the test's helper passed `last=None` for DSL incorrectly; it now builds the single-example dict explicitly. Second run: 23 passed.
- 16:21-16:23 `run.py --check-only`: **CHECK PASS** (full output below). All 512 DFA and 640 DSL mixes rebuilt with counts equal to `mixes.json`; no mix leaves an unused example in any target file, so all references are fresh draws (Section 2.2's fallback, seed 20261015); references and units disjoint; no task-token byte in any stripped serialization; graded construction passed on its first draw (mean rho 0.723 / 0.406 / 0.277 / 0.063 for r = 1, 2, 4, all; 8 rejections).
- 16:29 A stale CPU lock from an earlier run on the box (finished at 15:56) was replaced with this run's lock. `/proc/loadavg` about 3-5 with only this run's jobs.
- 16:30-16:47 Disclosed pilot `run.py --pilot` (seed 15909, probe 14101, lowest- and highest-K target per family, 5 graded runs) -> `expt_v1/results_pilot/` (962 s). Per-example gradients about 290 per second (DFA) and 180 per second (DSL); Task2Vec 90 s and 36 s per (target, probe); one graded run about 95 s on 4 workers. Projection: graded 160 runs 63 min, scoring 78 min, total about 2.4 h, inside the 4 h cap. Graded projection under 90 min, so the pre-declared fallback does not fire (8 seeds kept).
- 16:37-16:48 analyze.py and report.py smoke-tested on a synthetic results folder (random scores with a planted signal, built from the pilot's references; not kept): all gates computed, figure checked by eye (two labels moved). analyze.py takes about 11 min on one core.
- Before the freeze: PROTOCOL Section 2.8 (pilot disclosure and implementation choices) and the Status line; unit tests rerun, 23 passed.
- 16:52 Freeze commit aa35bfe2 ("Freeze Experiment 15 protocol"). Before it: origin/main fetched, no change to this folder.
- 16:52-17:52 Gated chunk 1 (`run.py --output results --stage all --resume --max-seconds 3300`; load average under 1 before the start). References rebuilt identically to the check (16:53-16:55). All 160 graded runs finished 16:55-17:45 (57-108 s each on 4 workers; load average about 5). With 149 s of budget left, the job pool still started its first 4 scoring jobs (a DFA probe job takes about 11 min), and the command was stopped at its 60-minute foreground limit at 17:52:47 before `run_state.json` was written. No part file was written, no result was read, and no process was left running.
- 17:54 **Operational fix (scheduling only, no result seen):** run.py's job pool now starts a scoring job only while at least 900 s of the chunk budget is left (`START_MARGIN_SECONDS`; the longest pilot job took 660 s), so a chunk ends inside its time limit. `results/run_state.json` was written by hand for chunk 1: 3,600 s (command start 16:52:47 to stop 17:52:47), counted against the 4 h cap.
- 17:54-18:35 Chunk 2 (`--resume --max-seconds 3000`): the 12 probe jobs (b1, d and their controls). DFA probe jobs took 1,330-1,350 s each (about twice the pilot's 660 s; load average 6-7 from other work on the box), DSL 200-220 s, graded 840-900 s. Stopped starting jobs when less than 900 s of budget was left; 2,458 s; cumulative 6,058 s of the 14,400 s cap.
- 18:50-19:30 Chunk 3 (`--resume --max-seconds 3300`): Task2Vec and non-probe jobs; 2,422 s; 5 jobs left when the 900 s margin stopped new starts. (Note: run.py stamps each `run_state.json` entry when it writes it, so the "started" field of chunks 2-4 and of the Eq. 2 rerun is each chunk's end time; the hand-written chunk 1 entry carries its start.)
- 19:30:49-19:49:41 Chunk 4 (`--resume --max-seconds 3300`): the last 5 jobs, the merge of all 212 scoring jobs, and the Eq. 2 check; 1,125 s.
- 19:50 First `analyze.py` stopped with a ValueError before computing any gate: the DSL label-blind embedding score b1 (and b1_perm, b1_inp) is exactly 1.0 on every unit, because every DSL pool input appears among the references, and min-max normalization of a constant is undefined. **Fix (Deviation 2, before any gate was computed):** `scores.minmax_or_zero` maps a score that is constant over the pool to 0 (scikit-learn's MinMaxScaler convention); analyze.py uses it and lists every case in `gates.json` under `constant_scores`; `minmax` itself still raises, as its tests expect. Gates 1, 2, 3 (calibrated) and 4b do not depend on the constant chosen. The Eq. 2 check had written NaN for the DSL rows for the same reason, so jobs.py's Eq. 2 check uses the same function and was rerun.
- Second `analyze.py` attempt stopped in `laws.percentile_ci` on a bootstrap with no finite draw (the constant DSL b1 again). Fix: it returns [nan, nan] (Deviation 3). No gate threshold, seed or estimator changed.
- 19:51-20:19 Eq. 2 check rerun (`run.py --stage eq2`, 1,683 s): 96 rows, none NaN; max abs difference 0.0024 (tolerance 0.01), pass. Meanwhile a provisional analysis on the old Eq. 2 file gave the gate verdicts below; the final analysis matched it.
- 20:19 Verification debug pass (steps 1-3): mixes, disjointness, token-free serializations, exact (a)/(a')/(b2) recompute and the b1 sample all exact (difference 0.0). The d and d_perm 10% sample failed the 1e-6 relative check: max relative difference 4.7e-5 and 7.1e-5, max absolute 3.2e-8 (DFA probe 14101); DSL 2e-7. The worst cases are cosines of float32 per-example gradients near 0 (about -9.6e-5), where a different batch padding changes float32 summation order. **Check-definition fix (Deviation 5):** for the probe-sample scores verify.py now measures the difference relative to the target's pool range, the scale min-max normalization divides by, with the same 1e-6 bar, and prints the absolute and plain relative differences too. First full run (20:31-20:41) failed on b1 under that scale: the recompute (float64) was compared with the float32 saved scores, a 3e-8 rounding gap against b1's narrow DFA pool range; the plain relative difference after float32 rounding was 0.0. The comparison now rounds the recompute to float32 first; d and d_perm scaled differences were 7e-8. Second full run (20:46-21:12) passed steps 1-6 (b1 0.0; d and d_perm 7.1e-8 of the pool range; Task2Vec sample 0.0; every alpha-hat exact; graded agreement within 0.0037 of Monte Carlo; 1,280 graded checkpoints replayed with difference 0.0) and failed in step 7 on `dsl|b1: Spearman differs`: both the main and the independent Spearman are NaN there (constant score), and NaN never equals NaN. The comparison now counts undefined-in-both as agreement.
- 21:27-21:52 Third full `verify.py --results results` (25 min): **PASS**. All steps as above, plus laws refit with independent_fit.py within 1e-4 and Spearman, slopes and calibrator within 1e-10; Eq. 2 check 0.0024. Log in `results/verify_log.json`.
- report.py writes RESULTS_alpha_recovery.md and alpha_recovery.png (checked by eye; panel C's legend moved off a point, panel E on a linear axis). The run machine's host name in manifest.json (this run and the pilot) is replaced by a neutral label, and run.py now writes that label. One unit test added for `minmax_or_zero`; 24 pass. bootstrap.npz (1.7 MB) is committed; the large files stay out of git with their SHA-256 in ARTIFACT_SHA256.json.
- No further runs. Disclosed: this change was made after the gate numbers were seen; it changes no score, estimator or gate.
- 20:22-20:31 Final `analyze.py --results results` (519 s): G1 NOT_ESTABLISHED, G2 SUPPORTED, G3 SUPPORTED, G4a SUPPORTED, G4b NOT_ESTABLISHED. Key numbers in RESULTS_alpha_recovery.md.
- Gated run wall time (run_state.json): 11,289 s (3.14 h) of the 14,400 s cap, counting chunk 1 at its full 3,600 s; analysis and verification extra.

<details><summary><code>run.py --check-only</code> output (10-19-2026 16:21)</summary>

```
[10-19-2026 16:21:44] refs: dfa 512 mixes rebuilt, counts equal mixes.json: True
[10-19-2026 16:22:16] refs: dsl 640 mixes rebuilt, counts equal mixes.json: True
[10-19-2026 16:23:06] graded construction attempt 0: mean rho by r {1: 0.7226113597318395, 2: 0.4063491062581118, 4: 0.2772769991724646, 'all': 0.0627622709129854} -> PASS
[10-19-2026 16:23:20] refs: done in 132 s; token-free {'dfa': True, 'dsl': True, 'graded': True}; disjoint {'dfa': True, 'dsl': True}
{
 "check": "PASS",
 "mix_check": {
  "dfa": {
   "n_mixes": 512,
   "counts_equal": true,
   "t_max_count_spread": 7
  },
  "dsl": {
   "n_mixes": 640,
   "counts_equal": true,
   "t_max_count_spread": 1
  }
 },
 "token_free": {
  "dfa": true,
  "dsl": true,
  "graded": true
 },
 "disjoint": {
  "dfa": true,
  "dsl": true
 },
 "references": [
  {
   "family": "dfa",
   "target": "dfa_n04_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t1",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t2",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t3",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t4",
   "unused_file_examples": 0,
   "refs_from_file": 0,
   "refs_fresh": 2048
  }
 ],
 "overlap": [
  {
   "family": "dfa",
   "target": "dfa_n04_t1",
   "exact_content_overlap": 0.029296875,
   "reference_coverage": 0.054931640625
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t2",
   "exact_content_overlap": 0.029684627757352942,
   "reference_coverage": 0.05437155330882353
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t3",
   "exact_content_overlap": 0.03282973345588235,
   "reference_coverage": 0.05922564338235294
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t4",
   "exact_content_overlap": 0.02737247242647059,
   "reference_coverage": 0.050795611213235295
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t1",
   "exact_content_overlap": 0.02730066636029412,
   "reference_coverage": 0.053696576286764705
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t2",
   "exact_content_overlap": 0.02961282169117647,
   "reference_coverage": 0.05651137408088235
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t3",
   "exact_content_overlap": 0.02712833180147059,
   "reference_coverage": 0.05243278952205882
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t4",
   "exact_content_overlap": 0.029900045955882353,
   "reference_coverage": 0.055161420036764705
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t1",
   "exact_content_overlap": 0.028435202205882353,
   "reference_coverage": 0.054888556985294115
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t2",
   "exact_content_overlap": 0.02926815257352941,
   "reference_coverage": 0.056353400735294115
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t3",
   "exact_content_overlap": 0.030819163602941176,
   "reference_coverage": 0.05705710018382353
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t4",
   "exact_content_overlap": 0.029871323529411766,
   "reference_coverage": 0.05576459099264706
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t1",
   "exact_content_overlap": 0.028937844669117647,
   "reference_coverage": 0.05441463694852941
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t2",
   "exact_content_overlap": 0.027501723345588234,
   "reference_coverage": 0.052720013786764705
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t3",
   "exact_content_overlap": 0.027185776654411766,
   "reference_coverage": 0.05257640165441176
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t4",
   "exact_content_overlap": 0.031479779411764705,
   "reference_coverage": 0.05656881893382353
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t1",
   "exact_content_overlap": 0.05412946428571429,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t2",
   "exact_content_overlap": 0.05078125,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t3",
   "exact_content_overlap": 0.05264136904761905,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t4",
   "exact_content_overlap": 0.06017485119047619,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t1",
   "exact_content_overlap": 0.05282738095238095,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t2",
   "exact_content_overlap": 0.04827008928571429,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t3",
   "exact_content_overlap": 0.052455357142857144,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t4",
   "exact_content_overlap": 0.04985119047619048,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t1",
   "exact_content_overlap": 0.050595238095238096,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t2",
   "exact_content_overlap": 0.04975818452380952,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t3",
   "exact_content_overlap": 0.04845610119047619,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t4",
   "exact_content_overlap": 0.04920014880952381,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t1",
   "exact_content_overlap": 0.048828125,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t2",
   "exact_content_overlap": 0.05022321428571429,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t3",
   "exact_content_overlap": 0.049479166666666664,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t4",
   "exact_content_overlap": 0.04975818452380952,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t1",
   "exact_content_overlap": 0.04836309523809524,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t2",
   "exact_content_overlap": 0.048642113095238096,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t3",
   "exact_content_overlap": 0.048549107142857144,
   "reference_coverage": 1.0
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t4",
   "exact_content_overlap": 0.050595238095238096,
   "reference_coverage": 1.0
  },
  {
   "family": "graded",
   "target": "dfa_n08_t2",
   "exact_content_overlap": 0.042182074652777776,
   "reference_coverage": 0.058539496527777776
  },
  {
   "family": "graded",
   "target": "dfa_n08_t3",
   "exact_content_overlap": 0.03260633680555555,
   "reference_coverage": 0.048909505208333336
  },
  {
   "family": "graded",
   "target": "dfa_n16_t1",
   "exact_content_overlap": 0.041259765625,
   "reference_coverage": 0.053168402777777776
  },
  {
   "family": "graded",
   "target": "dfa_n16_t2",
   "exact_content_overlap": 0.039876302083333336,
   "reference_coverage": 0.05295138888888889
  }
 ],
 "tv": [
  {
   "family": "dfa",
   "target": "dfa_n04_t1",
   "tv_inputs": 0.0221405029296875,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t2",
   "tv_inputs": 0.02178955078125,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t3",
   "tv_inputs": 0.0298004150390625,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n04_t4",
   "tv_inputs": 0.0243377685546875,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t1",
   "tv_inputs": 0.02264404296875,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t2",
   "tv_inputs": 0.0261688232421875,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t3",
   "tv_inputs": 0.0274505615234375,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n08_t4",
   "tv_inputs": 0.0213623046875,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t1",
   "tv_inputs": 0.0257110595703125,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t2",
   "tv_inputs": 0.0211639404296875,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t3",
   "tv_inputs": 0.0221710205078125,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n16_t4",
   "tv_inputs": 0.024078369140625,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t1",
   "tv_inputs": 0.018280029296875,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t2",
   "tv_inputs": 0.026214599609375,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t3",
   "tv_inputs": 0.0244293212890625,
   "over": "length"
  },
  {
   "family": "dfa",
   "target": "dfa_n32_t4",
   "tv_inputs": 0.0239410400390625,
   "over": "length"
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t1",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t2",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t3",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k2_t4",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t1",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t2",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t3",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k3_t4",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t1",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t2",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t3",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k4_t4",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t1",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t2",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t3",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k5_t4",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t1",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t2",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t3",
   "tv_inputs": 0.0,
   "over": "input word"
  },
  {
   "family": "dsl",
   "target": "dsl_k6_t4",
   "tv_inputs": 0.0,
   "over": "input word"
  }
 ],
 "graded": {
  "construction": [
   {
    "attempt": 0,
    "mean_rho_by_r": {
     "1": 0.7226113597318395,
     "2": 0.4063491062581118,
     "4": 0.2772769991724646,
     "all": 0.0627622709129854
    },
    "passed": true
   }
  ],
  "n_rejections": 8,
  "alpha_share_soft_correlation": 0.11100873749468448,
  "counts_spread": 6
 },
 "graded_distractors": [
  {
   "task": "dfa_n08_t2_r1_1",
   "r": "1",
   "K": 8,
   "acceptance": 0.5387,
   "agree": 0.8686,
   "agree_exact_uniform": 0.8687,
   "rho": 0.7371
  },
  {
   "task": "dfa_n08_t2_r1_2",
   "r": "1",
   "K": 8,
   "acceptance": 0.4992,
   "agree": 0.9782,
   "agree_exact_uniform": 0.9786,
   "rho": 0.9563
  },
  {
   "task": "dfa_n08_t2_r2_1",
   "r": "2",
   "K": 6,
   "acceptance": 0.5625,
   "agree": 0.5104,
   "agree_exact_uniform": 0.5089,
   "rho": 0.0204
  },
  {
   "task": "dfa_n08_t2_r2_2",
   "r": "2",
   "K": 8,
   "acceptance": 0.6685,
   "agree": 0.5755,
   "agree_exact_uniform": 0.5718,
   "rho": 0.15
  },
  {
   "task": "dfa_n08_t2_r4_1",
   "r": "4",
   "K": 6,
   "acceptance": 0.5551,
   "agree": 0.6637,
   "agree_exact_uniform": 0.6644,
   "rho": 0.3272
  },
  {
   "task": "dfa_n08_t2_r4_2",
   "r": "4",
   "K": 8,
   "acceptance": 0.47,
   "agree": 0.6449,
   "agree_exact_uniform": 0.6452,
   "rho": 0.29
  },
  {
   "task": "dfa_n08_t2_rall_1",
   "r": "all",
   "K": 6,
   "acceptance": 0.6844,
   "agree": 0.4503,
   "agree_exact_uniform": 0.4484,
   "rho": 0.0
  },
  {
   "task": "dfa_n08_t2_rall_2",
   "r": "all",
   "K": 8,
   "acceptance": 0.6104,
   "agree": 0.5837,
   "agree_exact_uniform": 0.5848,
   "rho": 0.1668
  },
  {
   "task": "dfa_n08_t3_r1_1",
   "r": "1",
   "K": 8,
   "acceptance": 0.4783,
   "agree": 0.8732,
   "agree_exact_uniform": 0.8744,
   "rho": 0.7455
  },
  {
   "task": "dfa_n08_t3_r1_2",
   "r": "1",
   "K": 8,
   "acceptance": 0.4959,
   "agree": 0.6496,
   "agree_exact_uniform": 0.6503,
   "rho": 0.2986
  },
  {
   "task": "dfa_n08_t3_r2_1",
   "r": "2",
   "K": 8,
   "acceptance": 0.4201,
   "agree": 0.7515,
   "agree_exact_uniform": 0.7521,
   "rho": 0.4962
  },
  {
   "task": "dfa_n08_t3_r2_2",
   "r": "2",
   "K": 7,
   "acceptance": 0.4942,
   "agree": 0.7113,
   "agree_exact_uniform": 0.7126,
   "rho": 0.4219
  },
  {
   "task": "dfa_n08_t3_r4_1",
   "r": "4",
   "K": 6,
   "acceptance": 0.3342,
   "agree": 0.5032,
   "agree_exact_uniform": 0.5033,
   "rho": 0.0
  },
  {
   "task": "dfa_n08_t3_r4_2",
   "r": "4",
   "K": 7,
   "acceptance": 0.5719,
   "agree": 0.4512,
   "agree_exact_uniform": 0.4513,
   "rho": 0.0
  },
  {
   "task": "dfa_n08_t3_rall_1",
   "r": "all",
   "K": 6,
   "acceptance": 0.6236,
   "agree": 0.4494,
   "agree_exact_uniform": 0.4501,
   "rho": 0.0
  },
  {
   "task": "dfa_n08_t3_rall_2",
   "r": "all",
   "K": 8,
   "acceptance": 0.5799,
   "agree": 0.5713,
   "agree_exact_uniform": 0.5713,
   "rho": 0.1539
  },
  {
   "task": "dfa_n16_t1_r1_1",
   "r": "1",
   "K": 16,
   "acceptance": 0.4551,
   "agree": 0.8586,
   "agree_exact_uniform": 0.8589,
   "rho": 0.7163
  },
  {
   "task": "dfa_n16_t1_r1_2",
   "r": "1",
   "K": 16,
   "acceptance": 0.4508,
   "agree": 0.9346,
   "agree_exact_uniform": 0.9349,
   "rho": 0.8687
  },
  {
   "task": "dfa_n16_t1_r2_1",
   "r": "2",
   "K": 15,
   "acceptance": 0.4501,
   "agree": 0.8129,
   "agree_exact_uniform": 0.8122,
   "rho": 0.6244
  },
  {
   "task": "dfa_n16_t1_r2_2",
   "r": "2",
   "K": 15,
   "acceptance": 0.3773,
   "agree": 0.7059,
   "agree_exact_uniform": 0.7066,
   "rho": 0.4066
  },
  {
   "task": "dfa_n16_t1_r4_1",
   "r": "4",
   "K": 13,
   "acceptance": 0.5084,
   "agree": 0.6307,
   "agree_exact_uniform": 0.6286,
   "rho": 0.2617
  },
  {
   "task": "dfa_n16_t1_r4_2",
   "r": "4",
   "K": 16,
   "acceptance": 0.435,
   "agree": 0.8368,
   "agree_exact_uniform": 0.8365,
   "rho": 0.672
  },
  {
   "task": "dfa_n16_t1_rall_1",
   "r": "all",
   "K": 10,
   "acceptance": 0.5267,
   "agree": 0.5196,
   "agree_exact_uniform": 0.5185,
   "rho": 0.0411
  },
  {
   "task": "dfa_n16_t1_rall_2",
   "r": "all",
   "K": 15,
   "acceptance": 0.5536,
   "agree": 0.5272,
   "agree_exact_uniform": 0.5262,
   "rho": 0.0579
  },
  {
   "task": "dfa_n16_t2_r1_1",
   "r": "1",
   "K": 16,
   "acceptance": 0.5452,
   "agree": 0.8137,
   "agree_exact_uniform": 0.8135,
   "rho": 0.6225
  },
  {
   "task": "dfa_n16_t2_r1_2",
   "r": "1",
   "K": 16,
   "acceptance": 0.5769,
   "agree": 0.9197,
   "agree_exact_uniform": 0.9193,
   "rho": 0.8358
  },
  {
   "task": "dfa_n16_t2_r2_1",
   "r": "2",
   "K": 15,
   "acceptance": 0.5498,
   "agree": 0.6976,
   "agree_exact_uniform": 0.6964,
   "rho": 0.3863
  },
  {
   "task": "dfa_n16_t2_r2_2",
   "r": "2",
   "K": 16,
   "acceptance": 0.5986,
   "agree": 0.8761,
   "agree_exact_uniform": 0.8763,
   "rho": 0.7449
  },
  {
   "task": "dfa_n16_t2_r4_1",
   "r": "4",
   "K": 14,
   "acceptance": 0.6143,
   "agree": 0.6723,
   "agree_exact_uniform": 0.6713,
   "rho": 0.3224
  },
  {
   "task": "dfa_n16_t2_r4_2",
   "r": "4",
   "K": 16,
   "acceptance": 0.4977,
   "agree": 0.6722,
   "agree_exact_uniform": 0.671,
   "rho": 0.345
  },
  {
   "task": "dfa_n16_t2_rall_1",
   "r": "all",
   "K": 14,
   "acceptance": 0.5013,
   "agree": 0.5414,
   "agree_exact_uniform": 0.5403,
   "rho": 0.0823
  },
  {
   "task": "dfa_n16_t2_rall_2",
   "r": "all",
   "K": 13,
   "acceptance": 0.5035,
   "agree": 0.4009,
   "agree_exact_uniform": 0.3997,
   "rho": 0.0
  }
 ],
 "graded_mixes": {
  "dfa_n08_t2_m1": {
   "alpha_share": 0.15,
   "alpha_soft": 0.2209,
   "alpha_agree": 0.5894
  },
  "dfa_n08_t2_m2": {
   "alpha_share": 0.15,
   "alpha_soft": 0.8697,
   "alpha_agree": 0.9349
  },
  "dfa_n08_t2_m3": {
   "alpha_share": 0.15,
   "alpha_soft": 0.4313,
   "alpha_agree": 0.7105
  },
  "dfa_n08_t2_m4": {
   "alpha_share": 0.4,
   "alpha_soft": 0.45,
   "alpha_agree": 0.7102
  },
  "dfa_n08_t2_m5": {
   "alpha_share": 0.4,
   "alpha_soft": 0.5181,
   "alpha_agree": 0.7592
  },
  "dfa_n08_t3_m1": {
   "alpha_share": 0.15,
   "alpha_soft": 0.2154,
   "alpha_agree": 0.5838
  },
  "dfa_n08_t3_m2": {
   "alpha_share": 0.15,
   "alpha_soft": 0.5938,
   "alpha_agree": 0.7972
  },
  "dfa_n08_t3_m3": {
   "alpha_share": 0.15,
   "alpha_soft": 0.3748,
   "alpha_agree": 0.6771
  },
  "dfa_n08_t3_m4": {
   "alpha_share": 0.4,
   "alpha_soft": 0.4462,
   "alpha_agree": 0.7062
  },
  "dfa_n08_t3_m5": {
   "alpha_share": 0.4,
   "alpha_soft": 0.5377,
   "alpha_agree": 0.7626
  },
  "dfa_n16_t1_m1": {
   "alpha_share": 0.15,
   "alpha_soft": 0.1921,
   "alpha_agree": 0.5949
  },
  "dfa_n16_t1_m2": {
   "alpha_share": 0.15,
   "alpha_soft": 0.8236,
   "alpha_agree": 0.9121
  },
  "dfa_n16_t1_m3": {
   "alpha_share": 0.15,
   "alpha_soft": 0.5377,
   "alpha_agree": 0.7691
  },
  "dfa_n16_t1_m4": {
   "alpha_share": 0.4,
   "alpha_soft": 0.4297,
   "alpha_agree": 0.714
  },
  "dfa_n16_t1_m5": {
   "alpha_share": 0.4,
   "alpha_soft": 0.6947,
   "alpha_agree": 0.848
  },
  "dfa_n16_t2_m1": {
   "alpha_share": 0.15,
   "alpha_soft": 0.185,
   "alpha_agree": 0.5505
  },
  "dfa_n16_t2_m2": {
   "alpha_share": 0.15,
   "alpha_soft": 0.7698,
   "alpha_agree": 0.8867
  },
  "dfa_n16_t2_m3": {
   "alpha_share": 0.15,
   "alpha_soft": 0.5048,
   "alpha_agree": 0.7443
  },
  "dfa_n16_t2_m4": {
   "alpha_share": 0.4,
   "alpha_soft": 0.4247,
   "alpha_agree": 0.6827
  },
  "dfa_n16_t2_m5": {
   "alpha_share": 0.4,
   "alpha_soft": 0.6698,
   "alpha_agree": 0.8377
  }
 },
 "work": {
  "main_targets": {
   "dfa": 16,
   "dsl": 20
  },
  "graded_targets": 4,
  "scored_units": {
   "dfa": 131072,
   "dsl": 20480,
   "graded": 131072
  },
  "probe_jobs": 12,
  "nonprobe_jobs": 40,
  "t2v_jobs": 160,
  "graded_runs": 160
 }
}
```

</details>
