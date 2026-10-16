# Checkpoint: held-out exposure and difficulty laws
**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/02_heldout_exposure_difficulty_laws/CKPT_laws.md>
**TLDR:** Ready to launch the complete prospective test; source data from Varun's private repository stay private.
Created: 10-16-2026
Last updated: 10-16-2026
## State
- Owner: Codex root; one source writer in canonical local repository.
- Experiment: experiments/02_heldout_exposure_difficulty_laws/expt_v1.
- Completed: frozen design, source implementation, exhaustive oracle/split check, synthetic fitter recovery, supervisor death self-test.
- Next: commit/push protocol; clone exact revision to owned SNAP path; launch all 200 trajectories.
- Execution: skampere2; 2 CPU threads; 5,400-second training budget; no provider charges.
- Remote path: /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/laws-10-16-2026.
- Session: codex-the-difficulty-confound-laws-heldouts.
- Watch: persistent streamed supervisor output; inspect job_status.json on completion/failure.
- Interpreter: /lfs/skampere2/0/brando9/uv_envs/veribench/bin/python.
- Publication: main protocol now; verified result branch, pull request, merge, artifact receipt and cleanup afterward.
- Private companion: expt_v1/source_data and results/private_math ignored; do not publish without explicit authorization.
- Continuation: read job_status.json and cells.json; preserve failed cells; never overwrite existing run.
## Log
- 10-16-2026: Located original laws/data, froze new independent holdouts; verified private/public repository visibility before staging copied data.
