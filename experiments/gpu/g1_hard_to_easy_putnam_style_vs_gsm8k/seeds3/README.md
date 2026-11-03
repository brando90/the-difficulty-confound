# g1 seeds3: the 3-seed check of the pilot (Sherlock, H100)

Design, prediction and readout: `../../CKPT.md`, section "g1 3-seed check". Runs: `python ../../common/runs.py list --tier seeds3 --plans g1` (16: E and H at 5e-6 and 2e-5 × seeds 0–2, format-only F at 5e-6 × seeds 0–2, base).

| file | what it does |
|---|---|
| `env.sh` | sourced: own wheel-only venv under `$SCRATCH/tdc_g1s3/venv` (python 3.12.1, torch 2.6.0+cu124), the pilot's data at `$SCRATCH/tdc_gpu/data` (read only, manifest hash checked), own HF cache |
| `submit_g1s3.sh` | login node, once: venv, model download, code and data checks, submits the 16-task H100 array and the report job |
| `g1s3.sbatch` | one array task = one run (`train.py --tier seeds3 --index $ID --save-final`), H100 only, 25 min, Slurm mail END,FAIL |
| `pack_results.sh` | `seeds3_report.py` and `$SCRATCH/tdc_g1s3/g1s3_results.tgz` (no weights); run at the end of every task and by the report job |
| `seeds3_report.py` | the pre-registered readout from the per-item files, plus the grader check against the pilot's base run |
| `regrade.py` | re-grades a run from its saved weights (only if the grader check fails) |

Locally, after fetching the results: `tar xzf g1s3_results.tgz && python seeds3_report.py --out-root runs --json r.json --md r.md` recomputes every number.
