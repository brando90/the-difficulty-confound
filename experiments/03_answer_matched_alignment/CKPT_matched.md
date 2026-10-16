# Matched-answer experiment checkpoint
Created: 10-16-2026 13:21 PDT
Last updated: 10-16-2026 13:41 PDT

## Goal
Test alignment versus carry demand with identical answer distributions, plus prospective held-out prediction.
## What ran
Source fd284d4; skampere2 session codex-difficulty-e03-numerical, child 1365659. Preflight and synthetic supervisor failure detection passed. First heartbeat: 7/200 complete.
## Results
Scientific results pending. 117 matched sums; 4,680 intervention examples; 1,872 test, 936 calibration, 936 probe; zero split overlap.
## Stuck / blocked
Nothing. Cluster health has percentage disk warnings (188 GiB local, 1421 GiB shared free); <2 GiB need fits. Unrelated Claude path warning does not affect CPU Python or Codex.
## Next steps
[p 10/10] Observe remote expt_v1/results/job_status.json through persistent SSH monitor; after completion inspect verification.json, publish all artifacts, remove owned clone.
## Brando to-do
None.

## Log
- 10-16-2026 13:21 PDT: protocol frozen; new seeds and sampling, same arithmetic reference. Canonical Mac /Users/brandomiranda/the-difficulty-confound; owned remote /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/matched-10-16-2026; cleanup owner master.
- 10-16-2026 13:39 PDT: three remote Codex profiles failed refresh without scientific work; master adopted numerical supervisor. START sent 1a14670035f2ac4f with all-four BCC verified.
