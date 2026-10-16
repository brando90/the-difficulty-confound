# Held-out exposure and difficulty laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/02_heldout_exposure_difficulty_laws/README.md>

**TLDR:** Complete: 200/200 fixed-skill addition trajectories. The combined extension misses its primary predictive-improvement criterion; the predeclared starting-difficulty-only ablation improves forecasts. The authorized original-data breadth test shows no improvement.

[Controlled results](RESULTS_heldout_laws.md) · [Actual-data companion](RESULTS_actual_math.md) · [Frozen protocol](PROTOCOL.md) · [Results ledger](results.md) · [Checkpoint](CKPT_laws.md)

Eight independently fitted seed blocks forecast unseen network widths, exposure mixtures, and questions. Joint error reduction with both covariates is 6.80% [-3.52, 14.80]; with difficulty alone it is 15.80% [10.27, 20.52] (95% paired seed bootstrap, 384 aggregate targets; p-val=n/a). These results transfer the alignment-law functional form to fixed addition, without validating the distinct reinforcement-learning utility law.

The public companion includes all 24 used mathematics observations, six example counts, and all predictions, as explicitly authorized by Brando on 10-16-2026. Brando subsequently authorized all remaining project artifacts without further confirmation. The [full original grid and metadata](expt_v1/source_data/) are now published too: 51 grid observations, of which the fixed-budget analysis uses 24. The other 27 observations were not added to any fit. Breadth is a proxy, not measured repeated exposure or starting difficulty.

## Reproduce the saved results

Install the [recorded dependencies](expt_v1/requirements.txt), then from the repository root:

```bash
python experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/verify_artifacts.py
python experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/reproduce_math.py
python experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/report.py --text-only
```

The full verification replays 240 weight files. The original same-host verification is retained in [verification.json](expt_v1/results/verification.json); the local table-only check is separate. [Transfer manifest](expt_v1/results/transfer_manifest.json) records original artifact hashes, including preserved execution-version reports and logs. Regenerating figures may change rendering bytes across platforms.

For fresh training use `run.py --output /path/to/new-run`, followed by `analyze.py --output /path/to/new-run`. The scripts refuse to overwrite inconsistent outcomes or forecasts. Execution records and the frozen protocol retain historical paths at source revision `337149b051256420e82f746712cf00a79316b682`; current canonical paths are under `experiments/archive/`.

## Complete source archive

The original [companion outputs and transfer receipts](expt_v1/results/private_math/) and [initial pre-publication report](PRIVATE_math_results.md) are also retained. Names containing `private`, and permission statements in historical records, describe their original state; the standing authorization in the [project README](../../../README.md#publication-authorization) supersedes those statements. The current scientific account is [RESULTS_actual_math.md](RESULTS_actual_math.md). Original data, numerical outputs, and provenance receipts remain byte-identical.
