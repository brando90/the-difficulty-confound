# Results: Experiment 22, does transfer to never-seen classes track measured shared structure on real data?

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/22_shared_structure_real_data/RESULTS.md>

Frozen protocol: [PROTOCOL.md](PROTOCOL.md) at a0a31d7a, Amendment 1 (platform moved to FarmShare L40S) at 51e614a7, which is the commit the grid ran from. Ran Oct 20-21 2026 on FarmShare: array 1860946, post job 1860947. Transformers only (a 2.7 M-parameter ViT on CIFAR-100 superclasses, an 8.3 M-parameter word-level text transformer on 20 Newsgroups top-level groups), both from scratch.

## Headline

On real data, transfer to never-seen classes is carried by shared structure, and the structure we measure explains it better than compressed size does. Vision, p = 0: the shuffled class hierarchy transfers nothing (-0.014 [-0.032, 0.006], P1 SUPPORTED), the real hierarchy transfers about half (0.487 [0.420, 0.555]). The centroid margin tracks it with Spearman rho 0.498 [0.324, 0.642], below the pre-registered 0.6 bar, so **P2 is NOT_SUPPORTED** at the frozen threshold. It beats gzip complexity by R-squared 0.211 [0.101, 0.310] (**P3 SUPPORTED**); gzip bits alone explain 0.011. The text arm (n = 16, secondary) passes all three: rho 0.800 [0.505, 0.946], R-squared difference 0.581 [0.230, 0.799].

Against the synthetic programme: Exp 17 (disjoint support) found zero-shot data on disjoint tokens buys nothing (D1 SUPPORTED) and alignment does not dominate complexity (E_far 0.73 < 1). Exp 20 (DFA LSTM) found complexity costs data (G1 SUPPORTED, M_K = 13) and moves the exponent (G3b SUPPORTED). Exp 16 (carry chains) has not been run. Exp 22 agrees with the synthetic picture that removing structure removes transfer (P1), and adds that on real data a structure measure outpredicts a complexity measure (P3). It does not reproduce a near-perfect structure-transfer link in vision: rho 0.50 is a moderate link, and real minus shuffled transfer is 0.501.

## Verdicts (pre-registered, p = 0, units averaged over seeds, unit bootstrap 95% intervals)

| prediction | vision (primary) | text (secondary) |
|---|---|---|
| P1 shuffled-map transfer within +-0.05 of 0 | **P1 SUPPORTED** (-0.014 [-0.032, 0.006], n = 100) | P1 SUPPORTED (0.022 [-0.131, 0.189], n = 16) |
| P2 Spearman rho(transfer, centroid margin) >= 0.6 | **P2 NOT_SUPPORTED** (0.498 [0.324, 0.642], n = 100) | P2 SUPPORTED (0.800 [0.505, 0.946], n = 16) |
| P3 R-squared(margin) minus R-squared(gzip bits) CI above 0 | **P3 SUPPORTED** (0.211 [0.101, 0.310]) | P3 SUPPORTED (0.581 [0.230, 0.799]) |

What counts against H_B (Protocol Section 3): shuffled-map transfer clearly above 0 (not seen), or real-map transfer tracking no measured structure while tracking complexity (not seen: gzip rho 0.070 in vision, 0.341 [-0.253, 0.740] in text). P2 missing its bar is a miss on strength of the link, not a contradiction.

## Tables (verbatim from `report.py`)

Runs: 154 (txt_real 60, txt_shuf 24, vis_real 50, vis_shuf 20); structure measures present: True

| phase | mean in-dist test acc | min | share best epoch = last | median train s |
|---|---|---|---|---|
| txt_real | 0.843 | 0.812 | 0.00 | 50 |
| txt_shuf | 0.761 | 0.702 | 0.00 | 48 |
| vis_real | 0.662 | 0.624 | 0.18 | 297 |
| vis_shuf | 0.483 | 0.415 | 0.05 | 250 |

## Pre-registered predictions (p = 0)

| arm | P1 shuffled-map transfer | real-map transfer | P2 Spearman rho (margin) | R² structure | R² complexity | R² diff [CI] | verdicts |
|---|---|---|---|---|---|---|---|
| text | 0.022 [-0.131, 0.189] (n=16) | 0.620 [0.470, 0.761] (n=16) | 0.800 [0.505, 0.946] (n=16) | 0.701 | 0.120 | 0.581 [0.230, 0.799] | P1 SUPPORTED, P2 SUPPORTED, P3 SUPPORTED |
| vision | -0.014 [-0.032, 0.006] (n=100) | 0.487 [0.420, 0.555] (n=100) | 0.498 [0.324, 0.642] (n=100) | 0.222 | 0.011 | 0.211 [0.101, 0.310] | P1 SUPPORTED, P2 NOT_SUPPORTED, P3 SUPPORTED |

## Secondary structure measures (real map, p = 0): Spearman rho with transfer

| arm | measure | rho [95% CI] | n |
|---|---|---|---|
| text | gzip_bits | 0.341 [-0.253, 0.740] | 16 |
| text | jaccard_top200 | 0.762 [0.317, 0.937] | 16 |
| text | margin | 0.800 [0.505, 0.946] | 16 |
| text | nc_share | 0.897 [0.672, 0.976] | 16 |
| text | pad | -0.597 [-0.889, -0.110] | 16 |
| text | margin, real + shuffled units pooled | 0.923 [0.818, 0.964] | 32 |
| vision | gzip_bits | 0.070 [-0.149, 0.287] | 100 |
| vision | margin | 0.498 [0.320, 0.644] | 100 |
| vision | nc_share | 0.523 [0.364, 0.654] | 100 |
| vision | pad | -0.330 [-0.514, -0.121] | 100 |
| vision | pixel_margin | 0.443 [0.268, 0.589] | 100 |
| vision | margin, real + shuffled units pooled | 0.839 [0.795, 0.871] | 200 |

## The p dial (share of held-out-class training items mixed in, fixed budget)

| arm | map / p | normalized transfer [95% CI] | held-out acc | in-dist acc |
|---|---|---|---|---|
| text | real/p=0 | 0.620 [0.469, 0.762] (n=16) | 0.620 | 0.848 |
| text | real/p=0.01 | 0.634 [0.500, 0.761] (n=16) | 0.629 | 0.848 |
| text | real/p=0.05 | 0.713 [0.585, 0.829] (n=16) | 0.675 | 0.847 |
| text | real/p=0.2 | 0.841 [0.750, 0.926] (n=16) | 0.750 | 0.845 |
| text | real/p=1 | 1.033 [0.962, 1.094] (n=16) | 0.844 | 0.826 |
| text | shuffled/p=0 | 0.022 [-0.135, 0.188] (n=16) | 0.263 | 0.786 |
| text | shuffled/p=1 | 1.037 [0.952, 1.124] (n=16) | 0.753 | 0.736 |
| vision | real/p=0 | 0.487 [0.419, 0.557] (n=100) | 0.353 | 0.673 |
| vision | real/p=0.01 | 0.508 [0.442, 0.575] (n=100) | 0.365 | 0.671 |
| vision | real/p=0.05 | 0.568 [0.502, 0.635] (n=100) | 0.403 | 0.671 |
| vision | real/p=0.2 | 0.741 [0.679, 0.802] (n=100) | 0.505 | 0.664 |
| vision | real/p=1 | 1.075 [1.025, 1.124] (n=100) | 0.675 | 0.631 |
| vision | shuffled/p=0 | -0.014 [-0.032, 0.006] (n=100) | 0.043 | 0.524 |
| vision | shuffled/p=1 | 1.165 [1.088, 1.244] (n=100) | 0.506 | 0.442 |

## Side by side with the synthetic experiments

| experiment | setting | structure vs complexity finding |
|---|---|---|
| 16 carry chains | arithmetic transformer | not run |
| 17 disjoint support | synthetic GINC, transformer | disjoint data buys nothing zero-shot (D1 SUPPORTED, excess 27.9 vs 2.8 bits/token); alignment does not dominate (E_far 0.73) |
| 20 DFA LSTM | synthetic DFAs | complexity costs data (G1 SUPPORTED, M_K = 13.0), moves the exponent (G3b SUPPORTED); alignment sets a floor (G6 SUPPORTED) |
| **22 real data (this)** | CIFAR-100 ViT, 20NG transformer | structure removed gives no transfer (P1 SUPPORTED); a structure measure beats a complexity measure (P3 SUPPORTED); structure-transfer link in vision moderate, rho 0.498, P2 NOT_SUPPORTED |

## Quality control and caveats

- **Under-training flags:** share of runs whose best epoch is the last evaluation: vis_real 0.18, vis_shuf 0.05, txt_real and txt_shuf 0.00. A fifth of the real vision runs were still improving at epoch 100, so vision transfer values are probably slightly low. The frozen budget was not changed.
- **Operational deviation (Amendment 1, allowed):** the array was packed as `--array=0-30%4` with 5 runs per task, because the FarmShare GPU QOS accepts at most 32 submitted tasks; 155 runs became 31 tasks. Grid, seeds, hyperparameters and analysis are unchanged. All 154 grid runs plus the feature extractor are present, no `.error` files.
- **Run-to-run determinism:** the post job replays two cells. The vision replay is bitwise identical in predictions. The text replay is not (in-distribution 0.8623 vs 0.8508, held-out 0.5821 vs 0.5665, best epoch 14 vs 8, within the 0.03 tolerance): Flash Attention is non-deterministic in backward, so single text cells carry that much noise.
- **`verify.py --stage local`: manifest, analysis recomputation and RESULTS.md match PASS; one check FAILS.** Manifest (166 files) PASS. `analysis.json` recomputed within 1e-9 PASS. RESULTS.md matches `analysis.json` PASS. **Structure recomputed within 1e-4 FAILS**: 42 of the 232 units' values differ from the cluster's `structure.json`, by up to 0.011 (mostly the text arm's margin, nc_share and logistic-regression proxy A-distance; vision proxy A-distance in a few units). The cluster's package versions were matched here (numpy 2.3.5, scipy 1.18.1, scikit-learn 1.9.1 from `exp22_venv_freeze.txt`) and the failure stays identical, so the numpy version is not the cause. Cause, isolated: scikit-learn's `max_features=100000` tf-idf cut ranks terms with an unstable `argsort`, and numpy's SIMD sort breaks the many ties at the cut differently depending on the CPU's instruction set (this machine has AVX2 and no AVX-512). The selected vocabulary changes with `NPY_DISABLE_CPU_FEATURES` here (two different vocabulary hashes) while data hashes, versions and thread counts do not matter. Vision units, which use no tf-idf, match to 1e-15 except one proxy A-distance (0.004). The structure file is therefore reproducible only on a CPU of the same instruction-set class as the cluster node; it could not be reproduced on a machine without AVX-512. The frozen tolerance was not loosened. Instead the whole analysis was rerun with the locally recomputed structure: every verdict is identical, and primary numbers move in the third decimal (text R-squared difference CI 0.2304 to 0.2300, 0.7986 to 0.7983); the largest shift is the secondary text proxy A-distance rho, -0.597 to -0.609. So no conclusion depends on which structure file is used.
- **Units within a split share one training run**, so unit bootstrap intervals understate between-run variance (disclosed in Protocol Section 4). Vision has 5 splits, text 4.
- **Seen data:** Exps 11-21 were disclosed as seen before freeze; the CPU code-path pilot used seed 22909, never reused.

## Artifacts

`expt_v1/results/` (154 run JSONs, `analysis.json`, `structure.json`, `MANIFEST.sha256`, replays, logs; the 44 MB feature file is not committed) and `ARTIFACT_SHA256.json` (every file's sha256, plus the fetched tarball `exp22_results.tgz`, sha256 6fcecf32b634f5968e4f230e01307fc42f2badcda511b23ce61ae87d2a5f752f, which matches the cluster's `.sha256`).
