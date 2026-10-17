# Checkpoint Log: `08_rlvr_verifiable_reward_utility_laws`

- **2026-10-16:** Protocol frozen at commit `95b9c80a5e871b4d0d512c92fb3a788d4f7fae0f`.
- **2026-10-16:** Completed all 200 supervised pretraining checkpoints + 200 pure 0/1 verifiable-reward RL (RLVR / leave-one-out GRPO, $G=16$) trajectories on `skampere2` (`elapsed_seconds = 114.65`).
- **2026-10-16:** Independent verification (`verify.py`) passed with `status = PASS` (`max_replay_bits_difference = 0.0`, 200 final + 200 warm checkpoints replayed, 3,456,000 item rows verified).
- **2026-10-16:** Generated `RESULTS_rlvr_utility.md`, `rlvr_utility_results.png`, `rlvr_utility_results.pdf`, and `ARTIFACT_SHA256.json`. Archived to `experiments/archive/08_rlvr_verifiable_reward_utility_laws/`.
