# Checkpoint: fixed-width addition exposure
Created: 10-16-2026 11:35 PDT
Last updated: 10-16-2026 11:44 PDT
## Goal
- Test exposure dependence in experiments/archive/01_fixed_width_addition_exposure/; publish README and all outcomes.
## What ran
- 32/32 cells on skampere2; 159.02 seconds; source db2dc6d; remote session finished.
## Results
- Long-carry accuracy 9.60% → 94.89%; recovered gap 94.34% [92.74,95.86], eight paired seeds.
- Strict criterion not established: no-carry accuracy 100% → 93.92%, a 6.08-point drop against tolerance 5.
- 512,000 training and 160,000 prediction rows verified; original source/config hashes match.
## Stuck / blocked
- Nothing; final artifact publication and remote temporary-checkout cleanup in progress.
## Next steps
- [p 10/10] Publish verified artifacts and confirm remote cleanup.
- Recompute: python experiments/archive/01_fixed_width_addition_exposure/expt_v1/analyze.py
## Brando to-do
- None.
## Log
- 10-16-2026 11:35 PDT -- Prepared protocol, code, complexity note, and results ledger before training.
- 10-16-2026 11:38 PDT -- Frozen source db2dc6d verified remotely, full run launched, start email delivered.
- 10-16-2026 11:44 PDT -- All 32 cells completed; independent artifact audit passed; strict criterion failed retention; archived completed experiment with original protocol preserved.
