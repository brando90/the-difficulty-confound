# Results: actual alignment-law breadth test

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/02_heldout_exposure_difficulty_laws/RESULTS_actual_math.md>

Source: `madanva/alignment-scaling-laws`, revision `7dfebb15fa4211a4784231e33e4976c6f86e65f6`.

**TLDR:** Adding log training-example count did not improve the original additive law on the saved mathematics grid. This tests a breadth/length proxy, not measured exposure or starting difficulty.

Executed on skampere2 on 10-16-2026. There are 24 observations: four model sizes, six training conditions, all at 21 million tokens. Original outcomes are retrospective and mostly single-seed. Joint folds withhold an entire size and condition (15 fitting cells); model-only and condition-only folds have 18 and 20 fitting cells. All transformations are fit within each fold. No full-data initializer is used. The baseline is loss = a + b N^(-beta) - d alpha, with model size N and alignment alpha; the constant token term is absorbed into a. The augmentation adds standardized log example count. Beta is selected within each fitting fold from [.05, .1, .25, .5, 1, 2] with bounded linear least squares, a declared adaptation of the original optimizer. This tests the published functional form on actual observations, not an exact reproduction of the original fitting procedure.

| Holdout | Baseline root mean squared error | With breadth | Relative error reduction, 95% descriptive interval | n |
|---|---:|---:|---:|---:|
| joint | 0.020380 | 0.020613 | -1.14% [-9.58, 0.57] | 24 |
| model_only | 0.023186 | 0.023278 | -0.40% [-2.16, -0.12] | 24 |
| mixture_only | 0.007392 | 0.007766 | -5.07% [-17.34, 1.35] | 24 |

Loss uses the original evaluation-loss units. Intervals: 20,000 paired bootstrap draws of four model-size blocks; overlapping fitted folds and only four blocks make these descriptive sensitivity intervals, not confirmatory population inference. p-val=n/a. A negative reduction means worse forecasts.

Alignment versus log breadth correlations are Pearson 0.977969, Spearman 1.000000, and Kendall tau-b 1.000000 across six unique conditions (repeated at four sizes); this describes six fixed conditions, with interval and p-val=n/a rather than population correlation inference. These features are confounded. No original per-question results or independently measured pre-update difficulty are available.

| Holdout | Law | Mean absolute error | Bias (prediction minus observation) | Prediction mean | Outcome mean |
|---|---|---:|---:|---:|---:|
| joint | additive | 0.015399 | 0.004806 | 0.749742 | 0.744936 |
| joint | breadth | 0.015473 | 0.004379 | 0.749316 | 0.744936 |
| model_only | additive | 0.018060 | 0.006765 | 0.751701 | 0.744936 |
| model_only | breadth | 0.018060 | 0.006765 | 0.751701 | 0.744936 |
| mixture_only | additive | 0.005567 | 0.000068 | 0.745004 | 0.744936 |
| mixture_only | breadth | 0.005744 | -0.000358 | 0.744578 | 0.744936 |

Brando explicitly authorized publication of this summary, the observations, and the predictions on 10-16-2026. The public input subset contains all 24 used observations and the six required example counts; unused rows and unrelated source metadata remain in the private source copy. [Inputs](expt_v1/public_math/source_data/) · [All predictions](expt_v1/public_math/results/actual_math_predictions.csv) · [Fitted coefficients](expt_v1/public_math/results/actual_math_fits.json).

Reproduce with `python experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/reproduce_math.py`. This invokes the frozen fitting/holdout implementation on the published input subset and checks its predictions against the archived original run. The original source-file hashes are recorded in dispatch.json and the original metadata receipt; the public subset has separate hashes. No prediction or outcome is omitted.

Public-subset reproduction passed for 144 predictions, three comparisons, and 144 fit records, with exact identities and maximum numerical difference 9.70e-14 (absolute and relative tolerances 1e-8). Standalone reproduction also passed without the private source files. [Provenance](expt_v1/public_math/provenance.json) · [Reproduction receipt](expt_v1/public_math/verification.json).
