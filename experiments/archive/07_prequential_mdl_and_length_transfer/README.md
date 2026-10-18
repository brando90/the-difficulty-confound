# Experiment 07: Carry-Marginal Matched Heterogeneous Pretraining, Remediation, and Prequential MDL

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/07_prequential_mdl_and_length_transfer/README.md>

**Status:** Complete and archived. Gates 1, 2a, 2b, and 3 are `SUPPORTED`. Gate 4 is `NOT_ESTABLISHED` because pooled cascading-two-carry accuracy is 81.09% [79.51, 82.88], below the prespecified 85% line. See [`RESULTS_mdl_history.md`](RESULTS_mdl_history.md) and [`PROTOCOL.md`](PROTOCOL.md).

## Replay (for Laolu / collaborators)

Training is already finished (200 trajectories). These commands re-check the saved run; they do not retrain.

```bash
python3 experiments/archive/07_prequential_mdl_and_length_transfer/expt_v1/test_laws.py
python3 experiments/archive/07_prequential_mdl_and_length_transfer/expt_v1/run.py --check-only
python3 experiments/archive/07_prequential_mdl_and_length_transfer/expt_v1/verify.py
```

`verify.py` compares its own source hash with the training receipt in `results/runtime.json`. A second look on 10-17-2026, on torch 2.7.0, matched greedy predictions for 200 final models, 200 warm models, and both test and calibration splits (600/600 prediction checks, 0 mismatches). The largest bit difference was 2.48e-5, so the frozen `<= 1e-6` assert fails on that torch build. The training receipt records torch 2.5.1+cu124 and `max_replay_bits_difference = 0.0`.
