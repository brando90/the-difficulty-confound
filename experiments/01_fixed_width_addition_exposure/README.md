# Fixed-width addition under controlled training exposure

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/01_fixed_width_addition_exposure/README.md>

**TLDR:** Test whether changing training coverage recovers most of a held-out carry-chain accuracy gap at fixed task and training budget. All outcomes, including failure of the criterion, will be retained.

## Goal and decision

The uncertainty is whether data exposure alone can account for most of a substantial apparent difficulty gap in a minimal neural learning setup. The [locked protocol](PROTOCOL.md) defines the gap, paired analysis, and prospective criteria. This is a synthetic mechanism test, not a measurement of how much real-world language-model difficulty is caused by data.

## Method

1. Enumerate all 500,500 unordered pairs from 000 through 999.
2. Stratify by the longest consecutive carry-out chain: zero, one, two, or three.
3. Freeze 1,000 test and 250 validation pairs per stratum; remove those unordered pairs from every training pool.
4. Train the same network in four data conditions for each of eight fixed seeds.
5. Evaluate the final checkpoint once; report exact sums and conditional predictive bits per answer.
6. Recompute results from per-item tables with paired seed-bootstrap intervals.

## Files

```text
PROTOCOL.md             frozen methods and decision criteria
COMPLEXITY.md           theoretical interpretation and limitations
results.md              live progress and final result links
CKPT_addition.md        resumable state and event log
expt_v1/config.json     numerical settings
expt_v1/run.py          data, checks, training, per-item predictions
expt_v1/analyze.py      independent scoring, uncertainty, plot
expt_v1/supervise.py    bounded remote execution and exit receipt
expt_v1/cc.md           continuation instructions
expt_v1/results/        measured artifacts, created by execution
```

## Dependencies and execution

Python, NumPy, PyTorch, and optionally Matplotlib; no pretrained weights, downloaded datasets, or model-provider credentials. Execution uses the Stanford Network Analysis Project cluster with two processor threads and no graphics processor. Estimated memory under 2 gigabytes, disk under 200 megabytes, runtime roughly 5–45 minutes with a 90-minute hard cap; estimates are unmeasured before launch.

The cluster health preflight reports low percentage-free storage, but available space is approximately 189 gibibytes locally and 1,485 gibibytes on shared storage, far above this bounded job's needs. No unrelated files are removed. The unused Claude-profile symlink warning is unrelated to this Python job.

## Status

| Phase | Status |
|---|---|
| Protocol and implementation | DONE |
| Exhaustive deterministic checks | PENDING |
| 32 training cells | PENDING |
| Full analysis and publication | PENDING |

[Live results](results.md) · [Complexity interpretation](COMPLEXITY.md)
