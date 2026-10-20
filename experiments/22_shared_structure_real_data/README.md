# Experiment 22: shared structure on real data

Does a transformer trained from scratch handle classes it never saw only when they share structure with what it did
see (Brando's H_B), on real data? CIFAR-100 superclasses (ViT) and 20 Newsgroups top-level groups (text transformer):
one member of every group is held out, the class hierarchy is either real or shuffled (structure removed, same
inputs), a share p of the held-out classes' training items is mixed back in at a fixed budget, and the shared structure
of every held-out class is measured from inputs alone. Design, predictions P1–P3 and checks: `PROTOCOL.md`.

- `expt_v1/run.py` — one run per (phase, split, p, seed); `--list <phase>`, `--runlist ... --shard i --nshards n`, `--pilot`
- `expt_v1/structure.py` — measured shared structure and complexity per unit -> `results/structure.json`
- `expt_v1/analyze.py` -> `results/analysis.json`; `expt_v1/report.py` prints the RESULTS tables
- `expt_v1/verify.py --stage remote|local`; `expt_v1/test_splits.py`
- `expt_v1/prep.py` + `data_sha256.json` — raw files -> prepared data, every hash pinned
- `launch/submit_exp22.sh` — the whole grid on Sherlock (prep on dev, GPU arrays, post job); run lists in `launch/runs/`

Local data: `EXP22_DATA=<folder> python expt_v1/prep.py --raw <raw folder>` (raw files: `launch/fetch_data.sh`).
