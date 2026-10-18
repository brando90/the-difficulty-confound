# Checkpoint Log: `09_headroom_bounded_rlvr_utility_law`

- **2026-10-18:** Exploratory refit of candidate utility laws on Experiment 08's published outcomes (`exploratory/refit_exp08.py`), disclosed in the protocol.
- **2026-10-18:** Protocol frozen at commit `d25e66919701464b9045dfca7b2c9e1126e2f55e`.
- **2026-10-18:** Smoke run (2 seeds, 20 RL updates) passed end to end on a Sherlock dev node; discarded.
- **2026-10-18:** Completed all 200 supervised pretraining checkpoints + 800 RLVR trajectories ($G \in \{4, 8, 16, 64\}$, 4,096 rollouts per update) on Sherlock (job `48325211`, CPU, `elapsed_seconds = 406.28`).
- **2026-10-18:** Independent verification (`verify.py`) passed with `status = PASS` (`max_replay_bits_difference = 0.0`, 800 final + 200 warm checkpoints replayed, 11,232,000 item rows verified, frozen forecasts reproduced).
- **2026-10-18:** Generated `RESULTS_headroom_bounded.md` and `ARTIFACT_SHA256.json`. Archived to `experiments/archive/09_headroom_bounded_rlvr_utility_law/`.
