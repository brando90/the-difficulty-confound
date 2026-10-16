# Checkpoint: fixed-width addition exposure
Created: 10-16-2026 11:35 PDT
Last updated: 10-16-2026 11:38 PDT
## Goal
- Test exposure dependence in experiments/01_fixed_width_addition_exposure/; publish README and all measured outcomes.
## What ran
- 500,500 oracle checks passed; supervisor failure detection passed; remote session codex-difficulty-addition-10-16 launched.
## Results
- Training launched; locked matrix is 8 seeds × 4 conditions = 32 cells.
## Stuck / blocked
- Nothing. Storage percentage warnings evaluated against <200 MB outputs; existing absolute free space is sufficient.
## Next steps
- [p 10/10] Observe supervisor completion, retrieve and independently verify all artifacts, publish outcome.
- Exact status: ssh skampere2 cat /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/addition-10-16-2026/experiments/01_fixed_width_addition_exposure/expt_v1/results/job_status.json
## Brando to-do
- None.
## Log
- 10-16-2026 11:35 PDT -- Prepared protocol, code, complexity note, and results ledger before training.
- 10-16-2026 11:38 PDT -- Frozen source db2dc6d verified remotely, full run launched, start email delivered.
