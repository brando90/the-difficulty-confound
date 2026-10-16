# Matched-answer experiment checkpoint
Created: 10-16-2026 13:21 PDT
Last updated: 10-16-2026 13:21 PDT

## Goal
Test alignment versus carry demand with identical answer distributions, plus prospective held-out prediction.
## What ran
Combinatorial feasibility only: 117 eligible sums, at least 60 pairs per carry stratum. No training.
## Results
None yet. Protocol and config frozen before measurement.
## Stuck / blocked
Nothing. Cluster health has percentage disk warnings (188 GiB local, 1421 GiB shared free); <2 GiB need fits. Unrelated Claude path warning does not affect CPU Python or Codex.
## Next steps
[p 10/10] Implement the frozen protocol on SNAP, verify invariants before training, commit source freeze, then execute all 200 cells and analyze.
## Brando to-do
None.

## Log
- 10-16-2026 13:21 PDT: protocol frozen; new seeds and sampling, same arithmetic reference. Canonical Mac /Users/brandomiranda/the-difficulty-confound; owned remote /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/matched-10-16-2026; cleanup owner master.
