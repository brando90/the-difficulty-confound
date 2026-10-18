# Experiment 08: Reinforcement Learning with Verifiable Exact-Sum Rewards (RLVR) and Utility Scaling Laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/08_rlvr_verifiable_reward_utility_laws/README.md>

**Status:** Complete and archived. Gate 1 (post-RLVR loss), Gate 3 (utility reversal), and Gate 4 (remediation) are `SUPPORTED`. Gate 2, the pooled additive utility law, is `NOT_ESTABLISHED` at **-15.78% [-35.91, 3.15]** (width 96 **+48.56%**, width 384 **-38.91%**). See [`PROTOCOL.md`](PROTOCOL.md) and [`RESULTS_rlvr_utility.md`](RESULTS_rlvr_utility.md).

## Replay

```bash
python3 experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/test_laws.py
python3 experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/verify.py
```

A 10-17-2026 replay of the 200 final models on the test split, torch 2.7.0, matched greedy predictions 200/200. The largest bit difference was 4.58e-5, above the frozen `<= 1e-6` assert. The archived receipt records `max_replay_bits_difference = 0.0`.
