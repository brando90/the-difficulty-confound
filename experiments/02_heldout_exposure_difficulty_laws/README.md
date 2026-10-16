# Held-out exposure and difficulty laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/02_heldout_exposure_difficulty_laws/README.md>

**TLDR:** Prospective fixed-skill addition test of exposure and pre-intervention difficulty as forecasting covariates, with whole network widths and training mixtures withheld. Status: protocol frozen; launch pending.

[Protocol](PROTOCOL.md) · [Live results](results.md) · [Checkpoint](CKPT_laws.md)

Run `python expt_v1/run.py --check-only`, `python expt_v1/analyze.py --self-test`, then `python expt_v1/supervise.py` on the designated execution host. The supervisor trains the entire declared set and fits/scorers frozen forecasts. Outputs are under `expt_v1/results/`; no provider calls are made.

An additional retrospective breadth-proxy test uses a private collaborator repository. Its copied inputs and derived outputs remain ignored and private. The public script supports `--private-math-only` when an authorized user supplies the two original source files; it does not publish them automatically.
