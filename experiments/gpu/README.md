# GPU plans: hard-to-easy (g1), cross-domain (g2), LLM elasticity (g3)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/gpu/README.md>

Status: three pre-registered plans, written 10-18-2026, ready to dispatch, not run. The code passed a CPU smoke test (toy data, tiny random Qwen2 model, data prep, training and every evaluation; the base, poetry, web-only and repeated-hard paths are not yet smoke-tested, see `CKPT.md`) and a planted-effect test (fake runs with known answers, recovered by all three analyses and confirmed by `verify.py`). Nothing here has touched a GPU yet: the pilot is the first GPU check.

## The three plans

| Plan | Question | Headline gate | Main tier alone |
|---|---|---|---|
| [g1 hard-to-easy](g1_hard_to_easy_putnam_style_vs_gsm8k/PLAN.md) | Does training only on Putnam-style (olympiad) maths cover GSM8K (Grade School Math 8K), or does each end need its own data? | GSM8K easy-only − hard-only ≥ 3 points and hard-only wins on MATH-500 levels 4–5: not a ladder | 16 runs, ~10.4 A100-hours |
| [g2 cross-domain](g2_cross_domain_math_vs_poetry/PLAN.md) | Does hard maths buy any poetry? | Transfer ratio of hard maths on held-out poetry R_H ≤ 0.10; harder maths no more general than easy (R_H − R_E ≤ 0.05) | 13 runs, ~8.4 |
| [g3 LLM elasticity](g3_llm_complexity_vs_alignment_elasticity/PLAN.md) | On a real model, does alignment (target share) move the data bill more than complexity (difficulty), as the paper claims, or less, as Exp 13-small found (E = 0.80)? | E = e_α/e_K > 1 in every seed (or < 1 in every seed: reversed) | 37 runs, ~15.8 |

All three fine-tune Qwen2.5-0.5B (Apache-2.0) for 40.0M tokens per run with seeds 0, 1, 2, and repeat at Qwen2.5-1.5B as the scale tier.

## Shared runs

`common/runs.py` is one registry for all three plans. A run's id depends only on its model, mixture, budget, learning rate and seed, so g1's easy-only arm, g2's easy-only arm and g3's (grade-school, α = 1) cell are one run, trained once. Dispatching `PLANS=g1,g2,g3` therefore costs 24.0 A100-hours for the main tier (49 runs), against 34.6 if the three were run separately.

| Tier | What | Runs (all three plans) | A100-hours | Purpose |
|---|---|---|---|---|
| pilot | 0.5B, seed 0, ~10M tokens; easy-only and hard-only at learning rates 5e-6, 1e-5, 2e-5; poetry-only and one g3 mixture at 1e-5; base | 9 | ~2.0 | picks the learning rate by a fixed rule; checks throughput and every evaluation |
| seed-0 first look | main tier, `SEEDS=0` | 17 | ~8.1 | a preview with no gates; every run is reused by main |
| **main** | 0.5B, seeds 0–2, 40.0M tokens | 49 (67 with `SECONDARY=1`) | **~24.0** (~30.9) | the gated results |
| scale | 1.5B, seeds 0–2 | 49 (67) | ~70 (~89) | replication at 3× parameters |

Costs are `python3 common/runs.py cost --tier <tier> --plans g1,g2,g3`: A100-80GB hours at 35% model FLOPs utilisation for training, plus a measured rule of thumb for generation. Wall-clock is cost divided by the number of GPUs, plus about 20 minutes of queue and start-up per job. The pilot measures real throughput; scale the later tiers by it.

**With single-digit Sherlock hours:** run the pilot (~2 h) on Sherlock, then the seed-0 first look of g1 alone (`PLANS=g1 SEEDS=0`, ~3.6 h) if hours remain. Run the main tier on SNAP (skampere1/2/3, 8 GPUs, ~3 h) or on the NAIRR allocation. Each plan also has a pilot of its own (g1 ~1.6 h, g2 ~0.4 h, g3 ~0.3 h) via `PLANS=gN`.

## Order (do not skip a step)

1. **Freeze.** Commit this folder unchanged; that sha is the freeze commit. Write it in `CKPT.md`.
2. **Data.** `sherlock_prep.sbatch` (or `MODE=prep` on SNAP) builds the pools, evaluation sets and `manifest.json` (counts, SHA-256 of each file, g3's complexity values K) and downloads the models. It refuses to overwrite an existing manifest.
3. **Pilot** at the freeze commit.
4. **Pilot disclosure.** `python3 common/runs.py select-lr --out-root $OUT --write` applies the pre-registered rule and writes `selected_lr` into `common/suite.json`. Paste its output and the measured tokens per second into `CKPT.md`, commit both: that sha is the disclosure commit, and all later tiers run at it. Nothing else may change in that commit.
5. **Seed-0 first look** (optional), then **main**, at the disclosure commit.
6. **Analysis and verification** (Section "Analysis"). Report only after `verify.py` prints `VERIFY PASS` for each plan.
7. **Scale**, only after main has been analysed.

## Dispatch

`<sha>` is the freeze commit for steps 2–3 and the disclosure commit afterwards. Every launcher refuses to run if the clone is not at exactly that commit or has modified tracked files.

### Sherlock

```bash
# once, on a login node
mkdir -p $SCRATCH/tdc_gpu && cd $SCRATCH/tdc_gpu && mkdir -p slurm_logs
git clone https://github.com/brando90/the-difficulty-confound.git
git -C the-difficulty-confound checkout --detach <sha>
module load python/3.12            # runs.py needs only the standard library
R=the-difficulty-confound/experiments/gpu

# step 2: data and models (CPU partition, ~1-2 h; builds the virtual environment the first time)
sbatch --export=ALL,TDC_COMMIT=<sha> $R/launch/sherlock_prep.sbatch

# step 3: pilot (after the prep job has finished)
TIER=pilot PLANS=g1,g2,g3
N=$(python3 $R/common/runs.py count --tier $TIER --plans $PLANS)
sbatch --array=0-$((N-1))%8 --export=ALL,TDC_COMMIT=<sha>,TIER=$TIER,PLANS=$PLANS $R/launch/sherlock.sbatch
python3 $R/common/runs.py status --tier $TIER --plans $PLANS --out-root $SCRATCH/tdc_gpu/runs

# step 4: select the learning rate, record it, commit (from a clone where you can commit), then check out the new sha
python3 $R/common/runs.py select-lr --out-root $SCRATCH/tdc_gpu/runs --write

# step 5: seed-0 first look, then main (same array command; SEEDS and SECONDARY must also go into count)
TIER=main PLANS=g1,g2,g3
N=$(python3 $R/common/runs.py count --tier $TIER --plans $PLANS --seeds 0)
sbatch --array=0-$((N-1))%8 --export=ALL,TDC_COMMIT=<sha>,TIER=$TIER,PLANS=$PLANS,SEEDS=0 $R/launch/sherlock.sbatch
N=$(python3 $R/common/runs.py count --tier $TIER --plans $PLANS)
sbatch --array=0-$((N-1))%8 --export=ALL,TDC_COMMIT=<sha>,TIER=$TIER,PLANS=$PLANS $R/launch/sherlock.sbatch

# scale tier: 80 GB cards and a longer limit
N=$(python3 $R/common/runs.py count --tier scale --plans $PLANS)
sbatch --array=0-$((N-1))%8 -C GPU_MEM:80GB --time=08:00:00 --export=ALL,TDC_COMMIT=<sha>,TIER=scale,PLANS=$PLANS $R/launch/sherlock.sbatch
```

Each array task trains one run of the registry (1 GPU, 8 CPUs, 64 GB, 4 h limit; the longest 0.5B run, with three generation checkpoints, takes about 1 h on an A100). Runs that are already DONE, or claimed by another job, exit at once, so resubmitting the same array after a failure is safe; it never silently retries a crashed run (its claim stays until you move it aside, as the log message says, and note it in `CKPT.md`). Logs: `slurm_logs/` and `runs/logs/status.tsv` (date, tier, plans, index, job, exit code, seconds). Jobs that land on a GPU older than an A100-class card exit with a compute-capability message; add a `-C` constraint from `sh_node_feat -p gpu`.

### SNAP

There is no Slurm partition for this group on SNAP; the launcher runs on one GPU node you are logged in to, inside tmux, one worker per free GPU (at least 30 GB free).

```bash
# log in once to skampere1 (or 2/3; mercury1/2 only for the 0.5B tiers). Never retry a failed login in a loop.
mkdir -p /dfs/scratch0/$USER/tdc_gpu && cd /dfs/scratch0/$USER/tdc_gpu
git clone https://github.com/brando90/the-difficulty-confound.git && cd the-difficulty-confound
git checkout --detach <sha>
MODE=prep TDC_COMMIT=<sha> bash experiments/gpu/launch/snap.sh $(hostname -s)
TDC_COMMIT=<sha> TIER=pilot PLANS=g1,g2,g3 bash experiments/gpu/launch/snap.sh $(hostname -s)
tmux attach -t tdc_gpu      # watch; detach with Ctrl-b d
# after the disclosure commit:
TDC_COMMIT=<sha2> TIER=main PLANS=g1,g2,g3 bash experiments/gpu/launch/snap.sh $(hostname -s)
```

To split one tier across two nodes, give each a range of registry indices: `... snap.sh skampere1 0 24` and `... snap.sh skampere2 25 48` (both with `BASE` on the shared `/dfs` path so they share one output folder). The claim folder stops two nodes from training the same run.

### Codex or another agent

Everything above runs without questions. Its only inputs are `<sha>`, `TIER`, `PLANS` and, optionally, `SEEDS` and `SECONDARY`. An agent should stop and report, rather than improvise, if a launcher refuses (wrong commit, missing data, compute capability) or a run writes `FAILED`.

## Analysis

```bash
OUT=$SCRATCH/tdc_gpu/runs DATA=$SCRATCH/tdc_gpu/data PY=$SCRATCH/tdc_gpu/venv/bin/python G=experiments/gpu
$PY $G/g1_hard_to_easy_putnam_style_vs_gsm8k/analyze.py --out-root $OUT --tier main
$PY $G/g2_cross_domain_math_vs_poetry/analyze.py --out-root $OUT --tier main
$PY $G/g3_llm_complexity_vs_alignment_elasticity/analyze.py --out-root $OUT --data-root $DATA --tier main
$PY $G/common/verify.py --plan g1 --tier main --out-root $OUT --report-dir $G/g1_hard_to_easy_putnam_style_vs_gsm8k/results_main
$PY $G/common/verify.py --plan g2 --tier main --out-root $OUT --report-dir $G/g2_cross_domain_math_vs_poetry/results_main
$PY $G/common/verify.py --plan g3 --tier main --out-root $OUT --data-root $DATA --report-dir $G/g3_llm_complexity_vs_alignment_elasticity/results_main
```

Each `analyze.py` writes `results_<tier>/results.json`, `RESULTS_draft.md` and its figure; `verify.py` checks every run's DONE hash and completeness, recomputes the headline numbers with independent plain-Python code and prints `VERIFY PASS` or `VERIFY FAIL`. `analyze.py` warns and labels its output "monitoring only" while runs are missing.

## Folder

| Path | What |
|---|---|
| `g1_*/`, `g2_*/`, `g3_*/` | `PLAN.md` (the pre-registration), `config.json` (arms), `analyze.py` |
| `common/suite.json` | models, data pools, evaluation sets, optimiser, tiers; `selected_lr` is filled in by the pilot disclosure commit |
| `common/runs.py` | the run registry: list, show, count, status, cost, select-lr |
| `common/prep_data.py` | builds pools and evaluation sets from Hugging Face, held-out splits, 13-word decontamination, manifest |
| `common/train.py` | trains one run (packed causal language modelling, exact token mixtures) and evaluates every checkpoint |
| `common/evals.py` | bits per byte, greedy maths generation with Math-Verify plus a fallback grader, HellaSwag |
| `common/results.py`, `common/verify.py` | shared statistics; independent recomputation |
| `common/fake_runs.py`, `common/smoke_test.sh` | planted-effect runs; the CPU end-to-end test |
| `launch/` | `sherlock_prep.sbatch`, `sherlock.sbatch`, `snap.sh`, `common_env.sh` |
| `CKPT.md` | freeze and pilot disclosure log |

Licences: Qwen2.5 models Apache-2.0; NuminaMath-CoT Apache-2.0; FineWeb-Edu ODC-By 1.0; Gutenberg Poetry Corpus CC0; GSM8K, MATH-500 and HellaSwag MIT.

## Relation to the CPU experiments

The synthetic counterparts run on CPU: Exp 13-small measured the within-family elasticity ratio (E = 0.80), and Exp 17 (AT-437) has hard-to-easy and disjoint-support arms on HMM concepts. g1 and g2 are the real-model versions of those arms, and g3 is the real-model version of Exp 13's ratio. Where a synthetic and a GPU result disagree, the report says so; neither overrides the other.

**TLDR-end:** [tdc: GPU plans] Three pre-registered, dispatch-ready GPU plans on Qwen2.5-0.5B/1.5B (hard-to-easy Putnam-style→GSM8K, cross-domain maths→poetry, and the LLM counterpart of Exp 13's elasticity ratio) sharing one run registry: pilot ~2 A100-hours, main ~24 for all three, with Sherlock and SNAP launchers, analysis scripts and independent verification, tested on CPU and on planted fake runs.
