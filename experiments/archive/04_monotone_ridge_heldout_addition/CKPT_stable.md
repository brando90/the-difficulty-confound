# Fresh predictive-law experiment checkpoint
Created: 10-16-2026 15:31 PDT
Last updated: 10-16-2026 16:57 PDT

## Goal
Repair extrapolation while holding exact addition fixed; run fresh held-out validation and draft the Sanmi-inspired blog.
## What ran
Source frozen; all preflight checks passed; 200/200 trajectories and 40 warm starts completed in 15.88 minutes on skampere2; all 1,521 artifacts transferred and verified on canonical Mac.
## Results
Repair criterion SUPPORTED (`legacy_difficulty` 1,935,297,769,746.72 → `stable_difficulty` 0.4835 bits/answer; 100.00% [99.90, 100.00] reduction). Starting-difficulty feature criterion NOT_ESTABLISHED (`stable_base` 0.5010 → `stable_difficulty` 0.4835 bits/answer; 3.48% [1.12, 6.24] reduction, below 10% threshold). Secondary ranking-reversal replication SUPPORTED (`e_low` 24.07 [22.80, 25.46] pp, `e_high` 24.48 [22.77, 26.06] pp). Full report: `RESULTS_stable.md`.
## Stuck / blocked
None.
## Next steps
Archived under `experiments/archive/04_monotone_ridge_heldout_addition/`; proceed to Experiment 05 (heterogeneous pretraining histories and strict no-harm retention) and Experiment 06 (compositional carry exposure with matched per-column marginals).
## Brando to-do
None.

## Log
- 10-16-2026 15:31 PDT: allocated experiments/04_monotone_ridge_heldout_addition. Canonical /Users/brandomiranda/the-difficulty-confound; planned owned clone /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/stable-10-16-2026; cleanup owner master. No provider-key calls; 90-minute training cap, expected 25–40 minutes, two CPU threads, no graphics processor.

- 10-16-2026 15:44 PDT: frozen revision 39cd0b78eea437f38b89cbf36928b8e0e9947613 pushed. Slow shared-filesystem clone interrupted before experimental work; complete node-local clone now /lfs/skampere2/0/brando9/difficulty-owned/stable-10-16-2026, same revision. Codex gpt-5.6-terra/high/full-access exited on invalid refresh token; personal Claude Sonnet/high/full-access exited on weekly limit. Both executor sessions confirmed gone. Master adopts ordinary-code numerical supervision under runbook recovery; no further language-model executor is needed for fixed commands, and no claim all accounts exhausted. No scientific attempt consumed. Remote policy refreshed to 2a53641. Old clone and transfer git directory are owned cleanup targets.

- 10-16-2026 15:45 PDT: synthetic independent fits/bounds, exhaustive oracle and fresh-split checks, and supervisor failure test PASS. Numerical supervisor 1982690, child 1982704, tmux codex-difficulty-e04-numerical. START Gmail receipt 1a146e35e41a2f5b. Watch status/heartbeat under expt_v1/results; supervisor polls child every five seconds, detects exit and 90-minute timeout, then analysis and verification. Node reboot does not automatically resume. Google companion 19yNEZTimoCPr9H4D_P-53nN83dBNFz_LY-aFdRa6_Bk created in ChatGPT, link viewing enabled; blog draft written.

- 10-16-2026 15:51 PDT: draft pull request https://github.com/brando90/the-difficulty-confound/pull/3 opened and attached. Google text readback exact; seven native headings and anyone-with-link reader permission verified. Removed owned interrupted DFS checkout and transfer .git copy after exact-revision comparison and git fsck; active node-local numerical checkout retained. Training 87/200 cells; final test outcomes remain unread by master.

- 10-16-2026 16:00 PDT: 200/200 trajectories completed in 15.88 minutes and all forecasts frozen. Original verifier stopped on aggregate root mean squared error equality: width-384 legacy law 2736924353206.2124 versus 2736924353206.213, absolute delta 0.00048828125, relative delta 1.784e-16, exactly one representable float step. Numerical diagnosis retained. Separate verify_numeric_recovery.py replays every frozen check, changing only that scalar equality tolerance to max(1e-12, eight float steps); original source, failed job_status and log preserved. No models retrained, fits changed, forecasts removed or scientific gates altered. An early report invocation encountered the absent verification receipt and produced no report; rerun only after recovered verification.
