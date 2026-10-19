# Experiment 09: Headroom-Bounded RLVR Utility Laws and Group-Size Interventions

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/09_headroom_bounded_rlvr_utility_law/README.md>

See [`PROTOCOL.md`](PROTOCOL.md) for the frozen prospective protocol and [`RESULTS_headroom_bounded.md`](RESULTS_headroom_bounded.md) for results. [`exploratory/`](exploratory/) holds the disclosed refit on Experiment 08's published outcomes that informed it. `PROTOCOL.md` was frozen before the archive move, so its relative links point at the old location; the current paths are [`../08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md`](../08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md) and [`../../ideas/COLLABORATION_ONBOARDING.md`](../../ideas/COLLABORATION_ONBOARDING.md).

**Replay environment.** `verify.py` checks replayed bits to `1e-6`, which holds only in the training environment: Python 3.12.1, `torch==2.5.1+cpu`, `numpy==1.26.4`, `scipy==1.13.1` (recorded in `expt_v1/results/runtime.json`). Other torch builds reproduce every greedy prediction but differ by about `1e-5` to `1e-4` bits, so the tolerance assert fails there (for example `7.6e-5` on torch 2.7.0 in the PR #10 replay). To replay exactly:

```bash
python3.12 -m venv .venv-exp09 && source .venv-exp09/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install numpy==1.26.4 scipy==1.13.1
```

```bash
python3 experiments/archive/09_headroom_bounded_rlvr_utility_law/expt_v1/test_laws.py
python3 experiments/archive/09_headroom_bounded_rlvr_utility_law/expt_v1/run.py --check-only
python3 experiments/archive/09_headroom_bounded_rlvr_utility_law/expt_v1/run.py --output <run dir>
python3 experiments/archive/09_headroom_bounded_rlvr_utility_law/expt_v1/analyze.py --output <run dir>
python3 experiments/archive/09_headroom_bounded_rlvr_utility_law/expt_v1/verify.py --output <run dir>
```
