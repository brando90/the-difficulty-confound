# Experiment 23: Experiment 13-small and Experiment 17b's coverage arms trained to convergence, 8 seeds

Same concepts, references, 354,240-parameter transformer, optimizer and α grid as Experiment 13-small and
Experiment 17b; every run trains in stages (2^20, 2^22, 2^24, cap 2^26 tokens) until a pre-registered stopping rule
says the grid has converged; 8 seeds; 256 + 160 runs. The design, hypotheses (G1-G3, Q / H1′ / H2′) and gates are in
[PROTOCOL.md](PROTOCOL.md), frozen before any job.

| file | what |
|---|---|
| `PROTOCOL.md` | frozen protocol |
| `config.json` | grid, stages, stopping rule, ε* ladder, FarmShare settings |
| `run23.py` | prep, staged training (resume-safe), stopping rule, aggregation, smoke, compute estimate |
| `analyze23.py` | Experiment 13's analysis (unchanged) with the ε* rule; Experiment 17b's analysis with 8 seeds |
| `verify23.py`, `report23.py` | verification; RESULTS.md |
| `prep/prep_manifest.json` | sha256 of every frozen reference file (the files themselves ship in the package, not in git) |
| `tests/` | CPU tests (`python -m pytest -q tests`; `E23_SLOW=1` adds the 10-minute Experiment 13 analysis check); `tests/dryrun_pipeline.py` runs the whole chain on a miniature grid |
| `farmshare/` | `env.sh`, `prepare.sh`, `smoke.sbatch`, `stage.sbatch`, `decide.sbatch`, `analyze.sbatch`, `submit_full.sh`, `make_package.sh` |

## Running on FarmShare

From the JupyterLab terminal (a compute node; sbatch works there):

```bash
mkdir -p ~/e23 && tar xzf e23_pkg_<commit>.tgz -C ~/e23       # gives ~/e23/code/experiments/...
cd ~/e23/code/experiments/23_exp13_full_training_8seeds
bash farmshare/prepare.sh                                       # venv ~/venvs/tdc23, prep check, CPU tests
sbatch farmshare/smoke.sbatch                                   # 2 cells, stage 0; prints tok/s and the estimate
bash farmshare/submit_full.sh --dry-run                         # prints the 17 sbatch calls
bash farmshare/submit_full.sh                                   # submits the chain (about 425 jobs incl. array tasks)
```

Outputs: `~/e23/out/{e13,e17b}` (per-run records, checkpoints, training state, decisions, STOP, aggregated CSVs,
analysis), logs in `~/e23/logs`, job ids in `~/e23/job_ids_<time>.txt`. The analyze job writes `RESULTS.md` and copies
the small outputs to `results/` in this folder for the commit. If a decide job fails (runs missing after a stage),
`bash farmshare/submit_full.sh --from-stage K [--exp e13|e17b]` resubmits from that stage; finished runs are skipped.

Watch: `squeue -u $USER -n e23_e13_s0,e23_e17b_s0` (etc.), `cat ~/e23/out/e13/decisions/stage_0.json`,
`sacct -X -n -j <array id> -o state | sort | uniq -c`.

## CPU smoke (build machine, 10-20-2026)

`run23.py smoke`: the two cells through stage 0 (2^20 tokens + 2^17-token cooldown), on a 4-core box shared with
other jobs (load 6-15). Both reproduce their source runs exactly at 2^20 (same stream, same weights):

| cell | excess at 2^20 (bits/token) | source run at 2^20 | after cooldown | train tok/s (loaded box) | wall |
|---|---|---|---|---|---|
| e13 seed 13101, n = 32 target 1, α = 0.4 | 0.8295198541 | 0.8295198541 (Exp 13-small) | 0.738 | 1,029 | 33 min |
| e17b seed 17101, cov32, α = 0 (4 targets) | 2.349 / 2.913 / 2.615 / 2.885 | identical (Exp 17b) | 2.160 / 2.810 / 2.537 / 2.828 | 1,634 | 40 min |

The e13 cell's validation monitor fell 1.044 → 0.849 bits from 2^19 to 2^20, so the stopping rule would continue it
(as expected: the source runs were not converged).

## Compute estimate

Per run, a grid that goes to the cap trains 2^26 tokens plus cooldowns (1.125 × each stage's tokens, ≈ 1.16 × 2^26
in all) and evaluates at 11 checkpoints + 4 cooldown ends. `run23.py estimate` with the build machine's unloaded
throughput (4,635 training tokens/s per single-thread run, ×1.2 for on-the-fly sampling; 25 s per evaluation of one
target incl. validation):

| grid stops after | T_stop | core-hours (both) | wall, at %24 + %16 | slowest array task |
|---|---|---|---|---|
| stage 0 | 2^20 | 57 | 0.8 h | 0.25 h |
| stage 1 | 2^22 | 173 | 1.8 h | 0.34 h |
| stage 2 | 2^24 | 607 | 5.1 h | 1.1 h |
| stage 3 (cap) | 2^26 | 2,312 | 17.5 h | 4.2 h |

Plus queue waits and the analysis job (about 1-2 h: Experiment 13's analysis twice and its rerun in verification).
The smoke's loaded-box numbers (1,331 tok/s, 92 s per evaluation) give 3.5× these figures (8,060 core-hours,
61 h wall at the cap); FarmShare's real per-core speed comes from `smoke.sbatch`, which prints the same table for its
node. Time limits per stage are 2 h, 3 h, 8 h and 24 h (capped to the partition's limit by `submit_full.sh`); a task
that runs out of time resumes from its last checkpoint inside the stage on resubmission. Disk: about 15 GB
(checkpoints 6.4 GB, training states 6.7 GB).
