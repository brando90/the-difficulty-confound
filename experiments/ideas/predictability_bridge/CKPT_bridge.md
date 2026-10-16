# Checkpoint: exposure-aware predictability bridge
**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/ideas/predictability_bridge/CKPT_bridge.md>
**TLDR:** Grounded connection to Varun's projects and verified existing-data feature export; no new fitted-law claim.
Created: 10-16-2026
Last updated: 10-16-2026
## Goal
- Relate controlled exposure effects to the existing predictive laws without conflating task complexity and model-relative difficulty.
## Completed
- Located Varun's rl-env-scaling and alignment-scaling-laws repositories and the implemented RL utility model in rl-4-code-verification.
- Wrote the extension, leakage controls, grouping, holdout design, and identifiability limits in README.md.
- Exported 128 seed × condition × carry-stratum rows with separate features and outcomes, source hashes, and deterministic checks.
- Repeated export into a temporary directory and confirmed byte-identical outputs; removed the temporary directory.
## Remaining
- No new training or predictive fit is running. Prospective scale/budget/exposure replication remains a proposal.
## Reproduce
- `python3 src/difficulty_confound/export_scaling_covariates.py` from repository root.
## Log
- 10-16-2026: Read actual project equations and data schemas; documented the existing alignment–capacity interaction caveat.
- 10-16-2026: Verified unchanged test identities, train/test disjointness, actual exposure counts, exact correctness, and absence of target columns from features.
