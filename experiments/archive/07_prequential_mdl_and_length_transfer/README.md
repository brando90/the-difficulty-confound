# Experiment 07: Carry-Marginal Matched Heterogeneous Pretraining, Remediation, and Prequential MDL

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/07_prequential_mdl_and_length_transfer/README.md>

**Status:** Scaffolded and preflight-verified (`test_laws.py` and `run.py --check-only` pass). Ready for execution and PR closure! See [`PROTOCOL.md`](PROTOCOL.md) and [`../ideas/COLLABORATION_ONBOARDING.md`](../ideas/COLLABORATION_ONBOARDING.md).

## Quickstart (for Laolu / collaborators)

```bash
# 1. Verify laws and dataset invariants without training
python3 experiments/07_prequential_mdl_and_length_transfer/expt_v1/test_laws.py
python3 experiments/07_prequential_mdl_and_length_transfer/expt_v1/run.py --check-only

# 2. Run all 200 trajectories (takes ~2 minutes on an 8+ core CPU server)
python3 experiments/07_prequential_mdl_and_length_transfer/expt_v1/run.py

# 3. Run analysis and independent weight-replay verification
python3 experiments/07_prequential_mdl_and_length_transfer/expt_v1/analyze.py
python3 experiments/07_prequential_mdl_and_length_transfer/expt_v1/verify.py
```
