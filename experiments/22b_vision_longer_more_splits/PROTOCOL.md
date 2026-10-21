# Protocol: Experiment 22b, vision shared-structure rerun with a longer budget and ten class splits (`22b_vision_longer_more_splits`)

Status: FROZEN 10-21-2026, before any Experiment 22b run on any cluster. After the freeze nothing in Sections 1–6 changes; any change goes in Section 9 with the reason and what had been seen.

## 1. Why

Experiment 22 (`22_shared_structure_real_data`, RESULTS at 6f66bd3f) found on CIFAR-100 with a ViT trained from scratch: P1 SUPPORTED (shuffled-map transfer −0.014), P3 SUPPORTED (R² structure − R² gzip = 0.211 [0.101, 0.310]), P2 NOT_SUPPORTED (Spearman ρ = 0.498, unit-bootstrap 95% interval [0.324, 0.642], against the pre-registered 0.6). Two limits of that run are the reason for 22b:

1. **Budget**: 18% of the real-map vision runs had their best validation epoch at the last evaluation (epoch 100): the models may still have been improving.
2. **Splits**: only 5 vision splits, each one training run, and the 20 units of a split share it, so the unit bootstrap understates the uncertainty. Resampling the five splits instead gives ρ = 0.498 [0.307, 0.664] (a check on the Experiment 22 data, run before this freeze and disclosed in Section 8).

22b asks one question: **with a 3× longer training budget and ten independent splits, does the vision Spearman ρ between normalized transfer and the centroid-margin structure measure reach the 0.6 bar?** It tests P1–P3 again on this larger, longer-trained vision grid. Text is not rerun (its P1–P3 were SUPPORTED with n = 16; the follow-up is about the vision link).

## 2. Design (everything not listed here is as in Experiment 22's PROTOCOL.md, Sections 2.1–2.6)

- **Data, models, hyperparameters**: CIFAR-100 and CIFAR-10 (same files, same hashes as `data_sha256.json`), same ViT (2.7 M parameters), AdamW, batch 256, peak lr 1e-3, weight decay 0.05, label smoothing 0.1, crop+flip augmentation, bf16 on an L40S. **Changed: 300 epochs (was 100), 15 warmup epochs (was 5, the same 5% of training), cosine to 0 over the 300; validation every 5 epochs.** The feature extractor (CIFAR-10 ViT, seed 22777, 60 epochs, 5 warmup) is retrained as in Experiment 22.
- **Ten splits.** Real map: split k holds out member k % 5 of every superclass; splits 0–4 are the Experiment 22 partition (member k in fine-class-id order), splits 5–9 a second partition in which each superclass's five members are visited in a seeded random order (`numpy.random.default_rng(23500 + 1)` per superclass in order), so every fine class is held out exactly twice, in two different training contexts, and the two partitions differ in every split (`test_splits.py` checks it). Shuffled map: split k uses its own random group assignment, `default_rng(23000 + k)`, k = 0…9 (new permutations, not Experiment 22's), same group sizes. A **unit** is (map, split, held-out fine class): 200 real units and 200 shuffled units at each p.
- **Grid**: p ∈ {0, 1.0} for both maps; seeds 23101 and 23202 (new; not used in Experiment 22 or its pilot); 10 splits × 2 p × 2 seeds × 2 maps = 80 runs, plus the feature extractor: 81 runs. p = 1.0 is the fully-seen reference; the intermediate p of Experiment 22 is not repeated (Experiment 22 settled the dial; P1–P3 are about p = 0). Validation items, budget and the p dial are built as in Experiment 22 (`splits.py`; for splits 0–4 of the real map the validation items are identical to Experiment 22's).
- **Outcome, structure and complexity measures**: normalized transfer, centroid margin (primary), nearest-centroid share, proxy A-distance, pixel margin and gzip bits, all as in Experiment 22 Sections 2.4–2.5; the structure of each unit is computed from the seen classes of its own split.

## 3. Pre-registered predictions (p = 0, units averaged over the two seeds)

All intervals resample **whole splits** with replacement (10 splits, 20 units each; 20,000 draws, analysis seed 29102227). The Experiment 22 unit-level interval is reported next to it for comparison.

- **P1.** Vision, shuffled map: mean normalized transfer within ±0.05 of 0 (200 units). SUPPORTED if |mean| ≤ 0.05, else NOT_SUPPORTED.
- **P2.** Vision, real map: Spearman ρ(transfer, centroid margin) over the 200 units ≥ 0.6. SUPPORTED if the point estimate is ≥ 0.6, else NOT_SUPPORTED. The split-level 95% interval is reported with it; "reaches the bar" in the answer to Section 1 means this point-estimate rule, and whether the interval contains 0.6 is stated beside it.
- **P3.** R²(transfer ~ centroid margin) − R²(transfer ~ gzip bits), both simple linear fits on the 200 real units: SUPPORTED if the split-level interval of the difference is above 0, CONTRADICTED if below 0, NOT_ESTABLISHED otherwise.

Reported without a threshold: real-map transfer at p = 0 and p = 1; real minus shuffled; ρ of the secondary measures; ρ with real and shuffled units pooled; ρ within splits 0–4 and within splits 5–9; class-level ρ (each fine class's two splits averaged, n = 100, interval over classes); the seed-to-seed Spearman of the unit transfers (and the mean-of-two-seeds reliability, Spearman–Brown) for both maps; per-split mean transfer; the p = 0 vs p = 1 contrast.

**Budget check (the reason for the longer run)**: the share of real-map vision runs whose best validation epoch is the last evaluation (epoch 300) must be ≤ 0.05 (`analysis.json` `qc_budget`). If it is above 0.05 the result is reported with that limit stated; the budget is not extended inside 22b.

## 4. Analysis

`expt_v1/structure.py` writes `results/structure.json` (200 real and 200 shuffled units) from inputs and the frozen features only; `analyze.py` writes `results/analysis.json` (per-run QC, unit table, P1–P3, the descriptive block, the p dial); `report.py` prints the tables that go verbatim into `RESULTS.md`. A comparison with Experiment 22 (same 100 real units of splits 0–4: Spearman between the two transfers, mean shift at 100 versus 300 epochs) is computed afterwards from Experiment 22's committed `analysis.json` and reported as descriptive.

## 5. Verification (`expt_v1/verify.py`)

As in Experiment 22 Section 5, for this grid: `--stage remote` (FarmShare post job, GPU): data hashes, every run of the three frozen run lists present, no `.error` file, one replay from scratch (the first vision cell) within 0.03 in in-distribution and held-out accuracy, structure and analysis recomputed, `MANIFEST.sha256` written. `--stage local` (here, after the fetch): manifest, `analysis.json` recomputed within 1e-9, structure recomputed within 1e-4 (on an AVX-512 node: the structure check of Experiment 22 depended on the CPU instruction set only for text; vision structure is dense linear algebra), verdicts and the two p = 0 means in `RESULTS.md` match. `test_splits.py` adds: both real rounds hold out every fine class exactly once, the second round's order differs, and the ten shuffled maps are pairwise different.

## 6. Compute

FarmShare L40S as in Experiment 22 Amendment 1: `launch/submit_exp22b.sh` (prep job on `normal` that reuses `~/venvs/exp22` and `~/exp22/{raw,data}`, then one GPU array of the 81 runs packed into at most 25 tasks, at most 4 GPUs at once, then a post job). Estimated cost: about 15–20 minutes per vision run (Experiment 22: median 297 s for 100 epochs), roughly 20–25 GPU-hours, 5–7 hours of wall time at 4 GPUs plus queue. Slurm mail END,FAIL to both of Brando's addresses. Transformers only (Brando, 10-20-2026).

## 7. Code-path pilot

CPU, tiny models, seed 22909 (`run.py --pilot`): the feature extractor and vision cells for real split 0 (p = 0 and 0.2), real split 5 (second round), shuffled splits 0 and 9; then `structure.py`, `analyze.py` and `verify.py --stage remote --pilot`. Its accuracies say nothing about P1–P3 and are not used. Result (10-21-2026, before the freeze): every cell ran, `structure.py` computed all 400 units, `analyze.py` and `report.py` produced every table (with a partial grid the split-level interval is skipped), `verify.py --stage remote --pilot` and `--stage local --pilot` both PASS, the replay reproduced the cell exactly. The analysis was also run on a copy of the Experiment 22 vision runs relabelled as ten splits: it reproduces Experiment 22's ρ = 0.4977 and R² difference 0.211 exactly.

## 8. Disclosures

- Experiment 22's results were seen before this design (that is the point of 22b). So was a descriptive check on them, made while writing this protocol: the seed-to-seed Spearman of unit transfers at p = 0 is 0.956 on the real map (0.765 shuffled), so seed noise cannot be what holds ρ at 0.5, which is why 22b adds splits and training time and not seeds; and the split-level interval for Experiment 22's ρ is [0.307, 0.664].
- Neither "300 epochs" nor "ten splits" was tuned on any 22b result: no 22b run exists. Experiment 22's curves (best epoch spread over 55–100, 18% at 100) motivated a 3× budget under a cosine schedule that ends at 0; whether that suffices is the budget check of Section 3.
- The p dial of Experiment 22 is not repeated; text is not rerun.
- Measured structure is close in spirit to a nearest-centroid classifier on frozen features, so P2 is partly a validity check of the measure; P1 (the shuffled-map control) is the causal test of H_B (as in Experiment 22).

## 9. Deviations

None yet.
