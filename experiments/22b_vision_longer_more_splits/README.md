# Experiment 22b: vision shared-structure rerun (300 epochs, ten splits)

Follow-up of `22_shared_structure_real_data`: same question (does transfer to never-seen CIFAR-100 classes track the measured shared structure with the centroid-margin measure, Spearman ρ ≥ 0.6?), with a 3× training budget, ten independent class splits (two member-order partitions of the real map, ten new shuffled maps) and split-level (cluster) bootstrap intervals. Design, predictions and checks: `PROTOCOL.md`.

- `expt_v1/` — code, a copy of Experiment 22's with the 22b changes (`config.json`, `splits.py`, `analyze.py`, `report.py`, `test_splits.py`, vision-only `run.py`/`structure.py`/`verify.py`)
- `launch/submit_exp22b.sh` — FarmShare submit (prep, one GPU array packed to fit the gpu QOS, post job); run lists in `launch/runs/`
