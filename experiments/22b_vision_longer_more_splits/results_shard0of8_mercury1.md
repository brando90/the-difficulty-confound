# Experiment 22b: partial results from mercury1 (AT-1211 weekend run, finished 10-25-2026 01:26 PDT)

**Status:** PARTIAL, 6 of the 81 frozen runs (feature extractor + shard 0 of 8 of the real-map vision runs, five cells). This is not an answer to the 22b question: P1-P3 and the Spearman rho against the 0.6 bar need all the runs (shuffled map, p = 1.0, other seeds), and the frozen plan is FarmShare L40S. Nothing in PROTOCOL.md was changed. Hardware differs from the frozen plan (RTX A4000, disclosed here).

**Takeaway:** the code path and 300-epoch budget run cleanly end to end on an A4000 (no .error files, all five cells reach epoch 300, 20-way group classification, chance 0.05). At p = 0 (held-out members never seen in training) the held-out member accuracy is 0.32-0.40 across the five real-map splits, against in-distribution test accuracy 0.67-0.70, so a transfer gap of about 0.3 is visible on every split. Five cells are too few to say anything about the structure measure.

| cell (spec) | split k | seed | in-dist test acc | held-out acc | best val acc (epoch) | train s |
|---|---|---|---|---|---|---|
| vis_real_k0_p0_s23101 | 0 | 23101 | 0.6659 | 0.3365 | 0.668 (160) | 6008 |
| vis_real_k2_p0_s23202 | 2 | 23202 | 0.7016 | 0.3410 | 0.689 (295) | 1600 |
| vis_real_k4_p0_s23101 | 4 | 23101 | 0.6735 | 0.3580 | 0.6765 (290) | 6003 |
| vis_real_k6_p0_s23202 | 6 | 23202 | 0.6769 | 0.3235 | 0.686 (245) | 1601 |
| vis_real_k8_p0_s23101 | 8 | 23101 | 0.6713 | 0.3955 | 0.666 (295) | 1891 |

Feature extractor (feat_cifar10_vit): CIFAR-10 ViT, 2,693,578 parameters, 60 epochs, test accuracy 0.8285, 1218 s; features sha256 train 05a84b29..., test 642685bd... (`cifar100_features_from_cifar10_vit.npz`, 44 MB, not committed).

Settings (from each result JSON and the frozen protocol): mercury1.stanford.edu, GPU 9 (NVIDIA RTX A4000 16 GB), torch 2.8.0+cu128, numpy 2.2.6, bf16 autocast, K = 20 groups, N_budget 38,000 train images, n_val 2,000, in-distribution test 8,000, p = 0.0, seeds 23101 and 23202 as in the frozen grid. Train time varies 1600-6008 s (k0 and k4 took about 6000 s, the others 1600-1900 s), most likely because the GPU was shared with the other AT-1211 jobs while they ran, so seconds are not comparable between cells.

Raw per-run JSONs (with per-epoch curves, per-class held-out accuracies, prediction hashes): experiments/22b_vision_longer_more_splits/expt_v1/results/{feat,vis_real}/. Remaining 75 runs: still to launch on FarmShare L40S (blocked until Sherlock/FarmShare sign-in returns Oct 26) or another empty GPU; no further runs started.
