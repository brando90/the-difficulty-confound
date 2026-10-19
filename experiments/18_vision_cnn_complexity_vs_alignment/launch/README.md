# Running Experiment 18 on Sherlock (CPU partition)

Ordinary batch jobs on the `normal` partition, one CPU each; no GPU hours, no interactive session. CIFAR-10 is the arm that needs this: about 15-25 min per run on one core, 204 runs (main grid 128 + hard-to-easy 76).

```bash
cd experiments/18_vision_cnn_complexity_vs_alignment
bash launch/setup_sherlock.sh                                  # once
VISION_RAW=$SCRATCH/exp18_raw VISION_DATA=$SCRATCH/exp18_npz bash launch/fetch_data.sh   # once
mkdir -p logs
for ph in cifar10_main cifar10_H; do
  $HOME/venvs/exp18/bin/python expt_v1/run.py --list $ph > runs_$ph.txt
  N=$(wc -l < runs_$ph.txt)
  sbatch --array=0-$((N-1))%200 --export=ALL,RUNLIST=runs_$ph.txt,NSHARDS=$N launch/sherlock_cpu.sbatch
done
```

Any other phase (`mnist_main`, `mnist_H`, `fashion_main`, `kmnist_main`, `fashion_H`, `kmnist_H`, `mis`) runs the same way with `-t 00:30:00`. Afterwards copy `expt_v1/results/` back (or commit it) and run `expt_v1/analyze.py`, `expt_v1/verify.py`, `expt_v1/report.py`.
