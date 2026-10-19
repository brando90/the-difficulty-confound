# Running Experiment 18 on Sherlock (CPU partition)

Every phase runs as ordinary batch jobs on the `normal` partition, one CPU per run; no GPU hours, no interactive session beyond submitting (PROTOCOL Sections 6.1 and 7). About 700 core-hours: 844 MNIST-family runs of about 25 min and 228 CIFAR-10 runs of about 100 min on one core.

```bash
mkdir -p $SCRATCH/exp18 && cd $SCRATCH/exp18       # put exp18_pkg.tar.xz here
tar xJf exp18_pkg.tar.xz && bash exp18_pkg/submit_exp18.sh 2>&1 | tee submit_exp18.log
```

`submit_exp18.sh` checks the package against its SHA256SUMS, then submits: `prep_sherlock.sbatch` (venv with CPU torch, raw data download, raw-file and decoded-array hashes against `expt_v1/data_sha256.json`; nothing else starts unless it passes), one array per phase from the frozen run lists in `runs/` (`sherlock_cpu.sbatch`, `%150` concurrent per phase; MNIST family `-t 3:00:00`, CIFAR-10 `-t 8:00:00`), and `post_sherlock.sbatch` after all of them (`verify.py --stage remote`, then `$SCRATCH/exp18/exp18_results.tgz` and its `.sha256`). Job ids go to `$SCRATCH/exp18/jobs_exp18.txt`; it refuses to submit twice. Finished runs are skipped on resubmission of a phase. Afterwards unpack `exp18_results.tgz` into `expt_v1/` and run `analyze.py`, `report.py` and `verify.py` locally.
