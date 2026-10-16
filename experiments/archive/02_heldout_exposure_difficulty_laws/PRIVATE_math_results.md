# Private companion: actual alignment-law breadth test

**Doc link:** Local private artifact; not published. Source: `madanva/alignment-scaling-laws`, revision `7dfebb15fa4211a4784231e33e4976c6f86e65f6`.

**TLDR:** Adding log training-example count did not improve the original additive law on the saved mathematics grid. This tests a breadth/length proxy, not measured exposure or starting difficulty.

Executed on skampere2 on 10-16-2026. There are 24 observations: four model sizes, six training conditions, all at 21 million tokens. Original outcomes are retrospective and mostly single-seed. Joint folds withhold an entire size and condition (15 fitting cells); model-only and condition-only folds have 18 and 20 fitting cells. All transformations are fit within each fold. No full-data initializer is used.

| Holdout | Baseline root mean squared error | With breadth | Relative error reduction, 95% descriptive interval | n |
|---|---:|---:|---:|---:|
| joint | 0.020380 | 0.020613 | -1.14% [-9.58, 0.57] | 24 |
| model_only | 0.023186 | 0.023278 | -0.40% [-2.16, -0.12] | 24 |
| mixture_only | 0.007392 | 0.007766 | -5.07% [-17.34, 1.35] | 24 |

Loss uses the original evaluation-loss units. Intervals: 20,000 paired bootstrap draws of four model-size blocks; overlapping fitted folds and only four blocks make these descriptive sensitivity intervals, not confirmatory population inference. p-val=n/a. A negative reduction means worse forecasts.

Alignment versus log breadth correlation is 0.977969 across six unique conditions (repeated at four sizes); no correlation inference is claimed. These features are confounded. No original per-question results or independently measured pre-update difficulty are available.

| Holdout | Law | Mean absolute error | Bias (prediction minus observation) | Prediction mean | Outcome mean |
|---|---|---:|---:|---:|---:|
| joint | additive | 0.015399 | 0.004806 | 0.749742 | 0.744936 |
| joint | breadth | 0.015473 | 0.004379 | 0.749316 | 0.744936 |
| model_only | additive | 0.018060 | 0.006765 | 0.751701 | 0.744936 |
| model_only | breadth | 0.018060 | 0.006765 | 0.751701 | 0.744936 |
| mixture_only | additive | 0.005567 | 0.000068 | 0.745004 | 0.744936 |
| mixture_only | breadth | 0.005744 | -0.000358 | 0.744578 | 0.744936 |

The source repository was verified PRIVATE and the destination PUBLIC. Raw inputs and all derived companion outputs remain ignored pending explicit publication authorization. The public prospective addition experiment is independent of this decision.
