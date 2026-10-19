# Running Experiment 18 on Sherlock (CPU partition)

Every phase runs as ordinary batch jobs on the `normal` partition, one CPU per run; no GPU hours, no interactive session beyond submitting (PROTOCOL Sections 6.1 and 7). About 700 core-hours: 844 MNIST-family runs of about 25 min and 228 CIFAR-10 runs of about 100 min on one core.

```bash
mkdir -p $SCRATCH/exp18 && cd $SCRATCH/exp18       # put exp18_pkg.tar.xz here
tar xJf exp18_pkg.tar.xz && bash exp18_pkg/submit_exp18.sh 2>&1 | tee submit_exp18.log
```

`submit_exp18.sh` checks the package against its SHA256SUMS, then submits: `prep_sherlock.sbatch` (venv `$SCRATCH/exp18_venv` from wheels only: Sherlock is glibc 2.17, so torch 2.6.0+cpu, numpy 2.2.6, scipy 1.16.3, pillow 12.2.0, pyarrow 20.0.0; raw data download, raw-file and decoded-array hashes against `expt_v1/data_sha256.json`; nothing else starts unless it passes), one array per phase from the frozen run lists in `runs/` (`sherlock_cpu.sbatch`, `%150` concurrent per phase; MNIST family `-t 3:00:00`, CIFAR-10 `-t 8:00:00`), and `post_sherlock.sbatch` after a successful prep and all arrays (`verify.py --stage remote`, then `$SCRATCH/exp18r2/exp18_results.tgz` and its `.sha256`). Job ids go to `$SCRATCH/exp18r2/jobs_exp18.txt`; it refuses to submit twice. Finished runs are skipped on resubmission of a phase. Afterwards unpack `exp18_results.tgz` into `expt_v1/` and run `analyze.py`, `report.py` and `verify.py` locally.


Resubmit (r2, Oct 19): the first submit (from `$SCRATCH/exp18`, jobs 48476434..48476477) failed in prep (pip built numpy from source; the assembler rejected `endbr64`), so no run started. r2 unpacks into `$SCRATCH/exp18r2` and leaves the old folder untouched.
