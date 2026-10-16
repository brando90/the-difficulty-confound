# Reproduce the released mathematics analysis

**Doc link:** [Public mathematics subset](https://github.com/brando90/the-difficulty-confound/tree/main/experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/public_math)

**TLDR:** These are the 24 mathematics observations and six training-subset counts actually used by the retrospective breadth analysis, with its original predictions, comparisons and fitted parameters. The reproduction uses the frozen analysis and configuration; it needs no training, model calls or private source files.

From the repository root, install NumPy and SciPy using the versions in `expt_v1/requirements.txt`, then run:

```sh
python experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/reproduce_math.py
```

The wrapper points `analyze.ROOT` at this directory, calls its unchanged `actual_data` function with the original `config.json`, and writes regenerated outputs into a temporary directory. It checks all 144 prediction rows, three comparison rows and 144 fit records against `results/`, preserving exact row order, keys, labels and counts. Numeric values must match with absolute and relative tolerances of `1e-8`; small numerical differences across supported SciPy/platform versions can occur. A mismatch fails the command. The mathematics computation imports only NumPy and SciPy from the experiment's dependency list.

`provenance.json` records the public subset hashes and the original full-source hashes. During publication preparation, the wrapper checked that the 24 rows exactly equal the original input filtered to 21 million tokens and that all six example counts match. Unused input observations and unused verification fields are excluded from this minimal reproduction subset. Following the broader authorization on 10-16-2026, the [complete original inputs](../source_data/) are published alongside it; the original 24-row analysis remains unchanged. `results/actual_math_metadata.json` preserves the original receipt unchanged, including its original full-source hashes; those hashes intentionally differ from the reduced public files. `verification.json` records the actual reproduction environment and numerical agreement. Re-running with `--write-receipts` updates the verification receipt, while existing provenance remains unchanged.

The baseline uses model parameter count and alignment; the augmented model also uses the logarithm of training-example count. All observations share a 21-million-token budget. Entire model sizes, training conditions, or both are excluded from each target's fit; exponent selection and breadth standardization use only that fit's training rows. This is a retrospective training-breadth proxy analysis over four sizes, not an independent measurement of starting difficulty or repeated exposure. The reported intervals resample only four model-size blocks and are descriptive.
