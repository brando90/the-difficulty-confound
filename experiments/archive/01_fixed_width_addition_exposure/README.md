# Fixed-width addition under controlled training exposure

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/01_fixed_width_addition_exposure/README.md>

**TLDR:** Balanced exposure recovered 94.34% [92.74, 95.86] of the baseline carry-chain gap, but the full criterion failed because no-carry accuracy dropped 6.08 points. All 32 runs and their per-item artifacts are preserved.

**Execution:** COMPLETE. **Scientific verdict:** strict prospective criterion not established. [Full report](RESULTS_addition_exposure.md).

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
RESULTS_addition_exposure.md  full report and limitations
results.md              final status and result links
CKPT_addition.md        resumable state and event log
expt_v1/config.json     numerical settings
expt_v1/run.py          data, checks, training, per-item predictions
expt_v1/analyze.py      independent scoring, uncertainty, plot
expt_v1/supervise.py    bounded remote execution and exit receipt
expt_v1/verify_artifacts.py  independent artifact audit
expt_v1/cc.md           continuation instructions
expt_v1/results/        committed tables, weights, traces, and figure
```

## Dependencies and execution

Python, NumPy, PyTorch, and optionally Matplotlib; no pretrained weights, downloaded datasets, or model-provider credentials. Execution uses the Stanford Network Analysis Project cluster with two processor threads and no graphics processor. Observed training duration was 159.02 seconds and artifacts occupy about 21 megabytes. The pre-run estimate was 5–45 minutes with a 90-minute hard cap; peak memory was not measured.

The cluster health preflight reports low percentage-free storage, but available space is approximately 189 gibibytes locally and 1,485 gibibytes on shared storage, far above this bounded job's needs. No unrelated files are removed. The unused Claude-profile symlink warning is unrelated to this Python job.

## Status

| Phase | Status |
|---|---|
| Protocol and implementation | DONE |
| Exhaustive deterministic checks | DONE |
| 32 training cells | DONE (32/32) |
| Full analysis and publication | DONE |

[Live results](results.md) · [Complexity interpretation](COMPLEXITY.md)
