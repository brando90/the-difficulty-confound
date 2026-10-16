# Checkpoint: held-out exposure and difficulty laws
**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/02_heldout_exposure_difficulty_laws/CKPT_laws.md>
**TLDR:** Running the complete prospective test; source data from Varun's private repository stay private.
Created: 10-16-2026
Last updated: 10-16-2026
## State
- Owner: Codex root; one source writer in canonical local repository.
- Experiment: experiments/02_heldout_exposure_difficulty_laws/expt_v1.
- Completed: frozen source 337149b051256420e82f746712cf00a79316b682; remote preflight passed; first 5/200 trajectories verified.
- Next: await complete set; verify all artifacts, score frozen forecasts, publish results.
- Execution: skampere2; 2 CPU threads; 5,400-second training budget; no provider charges.
- Remote path: /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/laws-10-16-2026.
- Session: codex-the-difficulty-confound-laws-heldouts.
- Watch: persistent execution session 37143 streams output; supervisor PID 2901902, training child 2902286.
- Interpreter: /lfs/skampere2/0/brando9/uv_envs/veribench/bin/python.
- Publication: main protocol now; verified result branch, pull request, merge, artifact receipt and cleanup afterward.
- Private companion: expt_v1/source_data and results/private_math ignored; do not publish without explicit authorization.
- Continuation: read job_status.json and cells.json; preserve failed cells; never overwrite existing run.
## Log
- 10-16-2026: Located original laws/data, froze new independent holdouts; verified private/public repository visibility before staging copied data.
- 10-16-2026: Started named supervisor; verified exact revision and useful training output; start email receipt 1a1462ae5f75471c. Private companion calculation completed remotely.
