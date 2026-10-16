# Continue the fixed-width addition experiment

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/01_fixed_width_addition_exposure/expt_v1/cc.md>

**TLDR:** Execute the frozen 32-cell addition exposure experiment, verify all per-item results, and publish the measured outcome. Success means complete execution and an honest prospective verdict, including a negative result. Preserve completed cells and the original protocol; do not tune on observed test scores.

SNAP PRE-APPROVAL (Brando, 2026-09-14): you run with full access (permission prompts and sandbox bypassed). Every command, install, Docker/GPU use, file and git operation (branch, commit, push, PR, merge per Rule 46) and every budgeted experiment call needed for this task is already approved. Never stop to ask for permission or confirmation; treat any approval step as granted, proceed, and log what you did. Pause only for a real blocker (missing credential, exhausted budget, a human-only login step).

Read ../PROTOCOL.md, ../CKPT_addition.md, and ../results.md. Numerical execution is ordinary Python on skampere2, with two processor threads, no graphics processor, no external model calls, and a 5,400-second cap. These are the frozen scientific and resource limits. The master owns repository edits and publication; the Python process owns its run outputs.

Run deterministic checks before training. Never rerun an existing output directory or silently discard a failed cell. After completion, analyze.py must reconcile all 32 cells, identical test rows, exact sums, and disjoint training manifests. Commit every per-item table used for analysis. Preserve checkpoint files externally with a manifest or transfer them to the canonical project. Report actual model architecture and library versions from runtime.json.

A long or dispatched run keeps results.md live in the repo — not a report at the end. The experiment's ../results.md must be updated and committed as meaningful results arrive; the dispatching master mirrors it if the numerical worker cannot push. Update ../CKPT_addition.md on launch, completion, or blockage. Send the authorized start/done notifications once each, with actual status.

**TLDR-end:** [the-difficulty-confound: experiments/01_fixed_width_addition_exposure/] Run the fixed Python experiment, retain all outcomes, independently verify the artifact tables, and publish the bounded conclusion.
