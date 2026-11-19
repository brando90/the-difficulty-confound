# Experiment 22b: vision shared-structure rerun (300 epochs, ten splits)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/22b_vision_longer_more_splits/README.md>

**TLDR:** Complete 80-run CIFAR-100 shared-structure rerun (300 epochs, 10 class splits) on FarmShare L40S: shuffled-map transfer stays at zero (`P1 SUPPORTED`, −0.002 [−0.015, 0.010] vs 0.491 [0.472, 0.509] real-map) and centroid-margin structure beats gzip complexity (`P3 SUPPORTED`, $R^2$ difference 0.214 [0.141, 0.294]), while Spearman $\rho$ = 0.488 [0.377, 0.595] misses the 0.6 bar (`P2 NOT_SUPPORTED`; independent mercury1 RTX A4000 80-run replicate gives the same verdicts at $\rho$ = 0.533 [0.409, 0.641]).

Follow-up of [`22_shared_structure_real_data`](../22_shared_structure_real_data/): same question (does transfer to never-seen CIFAR-100 classes track the measured shared structure with the centroid-margin measure, Spearman ρ ≥ 0.6?), with a 3× training budget, ten independent class splits (two member-order partitions of the real map, ten new shuffled maps) and split-level (cluster) bootstrap intervals.

Status: **COMPLETE** (primary FarmShare L40S run + independent mercury1 A4000 replicate). Primary results and verification: [RESULTS.md](RESULTS.md). Frozen design, predictions and checks: [PROTOCOL.md](PROTOCOL.md). Replicate summary and cross-cluster comparison: [results_mercury1_full.md](results_mercury1_full.md), [compare_runs.txt](compare_runs.txt).

- `expt_v1/` — code, a copy of Experiment 22's with the 22b changes (`config.json`, `splits.py`, `analyze.py`, `report.py`, `test_splits.py`, vision-only `run.py`/`structure.py`/`verify.py`); primary FarmShare outputs and `MANIFEST.sha256` in `expt_v1/results_farmshare/`, mercury1 outputs in `expt_v1/results/`
- `launch/submit_exp22b.sh` — FarmShare submit (prep, one GPU array packed to fit the gpu QOS, post job); run lists in `launch/runs/`
