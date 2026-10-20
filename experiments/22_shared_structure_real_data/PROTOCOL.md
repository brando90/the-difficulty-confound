# Protocol: does transfer to unseen classes track measured shared structure, on real data? (`22_shared_structure_real_data`)

Status: FROZEN 10-20-2026, before any Experiment 22 run on the cluster. Written after the code-path pilot of Section 7 (tiny models on CPU; it checks that every path runs and every metric computes, and its accuracies say nothing about the questions). After the freeze nothing in Sections 2–5 changes; any change goes in Section 9 with the reason and what had been seen.

## 1. Why

Experiments 11–21 (synthetic DFA, DSL, GINC and pair-image tasks) found that complexity is real but that the alignment coefficient does not dominate it, that Task2Vec does not track exact complexity, and that Varun's α does not recover the true alignment on the toys. Brando's objection (10-20-2026): in machine learning a model cannot do well on data it was not trained on unless there is some intersection, compositionality or shared structure between the training data and the evaluation data, and synthetic tasks may simply not be a realistic case. Call this **H_B: no shared structure, no transfer**.

This experiment tests H_B on two real datasets, with transformers trained from scratch, by (a) removing the shared structure while keeping the very same inputs (a shuffled class hierarchy), (b) measuring the shared structure of each held-out class from inputs alone and asking whether it predicts how well that class is handled, and (c) comparing that measure with an input-complexity proxy, as Experiments 11–21 compared alignment with complexity.

## 2. Design

### 2.1 Data

- **Vision: CIFAR-100**, 32×32 colour, 100 fine classes in 20 superclasses of 5 (the dataset's own hierarchy). 500 training and 100 test images per fine class. Raw files: the Hugging Face `uoft-cs/cifar100` parquet files (same images and labels as the Toronto release).
- **Text: 20 Newsgroups** (by-date split, the figshare archive scikit-learn uses), the 17 subgroups under four top-level hierarchies: comp (5 subgroups), rec (4), sci (4), talk (4). alt.atheism, misc.forsale and soc.religion.christian have no sibling under their top level and are left out. Headers, signature blocks and quoted lines are removed with scikit-learn's rules (copied into `expt_v1/data.py`, so the result does not depend on the installed version); documents with fewer than 5 word tokens after that are dropped (367 of 9,650 training and 254 of 6,425 test documents), leaving 9,283 training and 6,171 test documents.
- **CIFAR-10** (Hugging Face `uoft-cs/cifar10`) is used only to train the frozen feature extractor of Section 2.5. Its 10 classes are disjoint from CIFAR-100's by construction.
- Every raw file and every prepared array is pinned by SHA-256 in `expt_v1/data_sha256.json`; the cluster prep refuses to submit on any mismatch.

### 2.2 Maps, held-out classes, splits

A **base class** is a CIFAR-100 fine class or a 20NG subgroup. A **map** assigns base classes to groups, and the model's label is the group.

- **real**: the dataset's own hierarchy (20 superclasses; comp/rec/sci/talk).
- **shuffled** (the control: structure removed, inputs and group sizes unchanged): base classes assigned to groups of the same sizes by a fixed random permutation (`numpy.random.default_rng(22000 + k)` for vision, `22500 + k` for text, where k is the split).

**Split k** holds out member k of every group (members in the map's order: fine-class id order for the real CIFAR-100 map, the order listed in 2.1 for the real 20NG map, permutation order for shuffled maps). Vision splits k = 0…4 cover every fine class exactly once under the real map; text splits k = 0…3 (comp.windows.x is never held out under the real map). The held-out classes' test items are the evaluation of transfer; their training items are kept out of training except as Section 2.3 allows.

A **unit** is (arm, map, split, held-out base class): 100 vision units and 16 text units per map.

### 2.3 The p dial (direct exposure at a fixed budget)

**Validation**: a fixed share of each *seen* base class's training items (25 images per fine class; 10% of a subgroup's documents), drawn with a seed that depends on (arm, map, split) only. It is the same for every p and seed of a split and never contains held-out classes.

**Budget** N = the seen classes' remaining training items (vision 38,000 images; text 6,307–6,505 documents depending on the split and map). At share p, n_h = round(p × held-out classes' training items) held-out items are mixed in (stratified over held-out classes) and the seen pool is subsampled to N − n_h (stratified), so every run of a split trains on exactly N items. Grid: p ∈ {0, 0.01, 0.05, 0.2, 1.0} for the real map and p ∈ {0, 1.0} for the shuffled map. p = 0 is the never-seen condition; p = 1.0 is the fully-seen reference (all held-out training items in, still evaluated on their test items only).

### 2.4 Outcome

For each run and held-out class h: acc_h = share of h's test items assigned to h's group. In-distribution accuracy acc_in = accuracy on the test items of the seen classes. **Normalized transfer** T_h = (acc_h − 1/K) / (acc_in − 1/K), K = 20 (vision) or 4 (text): 0 is chance, 1 means a never-seen class is handled as well as the classes seen in training. Units are averaged over seeds before any statistic.

Accuracies are read from the weights of the epoch with the best in-distribution **validation** accuracy (ties: earlier epoch). Held-out accuracy is logged at every evaluation for the learning curves but never used for selection.

### 2.5 Measured shared structure and complexity (inputs only, computed before any accuracy is analysed)

For unit (h, its group g): C_j = mean feature vector of the seen members of group j (training items); c_h = mean feature vector of h's training items.

- **Primary, centroid margin** = cos(c_h, C_g) − mean over j ≠ g of cos(c_h, C_j). Vision features: the [CLS] output of a ViT (same architecture as 2.6) trained 60 epochs on CIFAR-10 only, then frozen, standardized over the CIFAR-100 training images. Text features: TF-IDF of unigrams and bigrams (our tokenizer, min_df 3, at most 100,000 terms, sublinear tf), fit on the training documents of all 17 subgroups.
- Secondary: **nearest-centroid share** (share of h's items whose nearest seen-group centroid, by cosine, is g); **proxy A-distance** 2(1 − 2·err) of a 2-fold logistic regression telling h's items from g's seen members' items (higher = less shared); vision **pixel margin** (centroid margin on 8×8×3 average-pooled pixels, model-free); text **top-200 Jaccard** (overlap of the 200 highest-χ² terms of h versus the other groups' seen subgroups and of g's seen subgroups versus the same).
- **Complexity proxy** (the Experiments 11–21 comparison): mean gzip size in bits (level 9) of h's test items, one item at a time (raw uint8 image bytes; document text).

These are measurements, not models under study; the logistic regressions in the proxy A-distance are the standard estimator.

### 2.6 Models and training (transformers only, trained from scratch)

- **Vision: ViT**, 4×4 patches (64 tokens + [CLS]), width 192, depth 6, 3 heads, MLP 768, pre-LayerNorm, learned positions, no dropout (2.7 M parameters). Random 32×32 crops from 4-pixel zero padding and horizontal flips. AdamW (β 0.9, 0.999), weight decay 0.05 on weight matrices only, peak learning rate 1e-3, 5 warmup epochs then cosine to 0, 100 epochs, batch 256, label smoothing 0.1, gradient clipping 1.0. Validation every 5 epochs.
- **Text: word-level transformer encoder**, vocabulary = the 20,000 most frequent words (document frequency ≥ 2) of **the run's own training documents** (so a never-seen subgroup's words that occur nowhere else are unknown tokens, as they would be to a model that never saw them), [CLS] + the first 255 tokens, width 256, depth 4, 4 heads, MLP 1024, dropout 0.1, pre-LayerNorm, learned positions (8.3 M parameters, 5.1 M of them the embedding table). AdamW, weight decay 0.05, peak learning rate 5e-4, 1 warmup epoch then cosine, 30 epochs, batch 64, gradient clipping 1.0. Validation every epoch.
- Initialization depends on the seed alone, so all runs with one seed start from the same weights within an arm. fp16 autocast with loss scaling on GPU; deterministic algorithms requested (GPU results are still not bit-reproducible across devices; Section 5 uses a tolerance).

### 2.7 Grid and seeds

| Phase | Runs |
|---|---|
| feat: CIFAR-10 ViT feature extractor, seed 22777 | 1 |
| vis_real: 5 splits × 5 p × seeds 22101, 22202 | 50 |
| vis_shuf: 5 splits × p ∈ {0, 1} × 2 seeds | 20 |
| txt_real: 4 splits × 5 p × seeds 22101, 22202, 22303 | 60 |
| txt_shuf: 4 splits × p ∈ {0, 1} × 3 seeds | 24 |

155 runs. Pilot seed 22909 (never reused). Analysis seed 29102226, 20,000 bootstrap draws.

## 3. Pre-registered predictions (if H_B is right)

All at p = 0, units averaged over seeds, unit bootstrap 95% intervals.

- **P1 (structure removed → no transfer).** Vision, shuffled map: mean normalized transfer within ±0.05 of 0. SUPPORTED if |mean| ≤ 0.05, else NOT_SUPPORTED. Text reported the same way as secondary (16 units).
- **P2 (transfer tracks measured structure).** Vision, real map: Spearman ρ(T_h, centroid margin) over the 100 units ≥ 0.6. SUPPORTED if the point estimate is ≥ 0.6, else NOT_SUPPORTED. Text reported as secondary (n = 16, low power).
- **P3 (structure beats complexity).** Vision, real map: R²(T_h ~ centroid margin) − R²(T_h ~ gzip bits), both simple linear fits. SUPPORTED if the bootstrap 95% interval of the difference is above 0, CONTRADICTED if below 0, NOT_ESTABLISHED otherwise. Text secondary.

Reported without a pre-registered threshold: real-map transfer at p = 0 (is it far above 0, i.e. does the real hierarchy carry structure that transfers?), real minus shuffled at p = 0, the secondary measures' ρ, ρ with real and shuffled units pooled, and the p dial (how fast transfer rises with direct exposure).

What would count against H_B: shuffled-map transfer clearly above 0 (models handle never-seen classes with no shared structure to the labels), or real-map transfer that does not track any of the measured structure while tracking complexity.

## 4. Analysis

`expt_v1/structure.py` writes `results/structure.json` (Section 2.5) from inputs and the frozen features only. `expt_v1/analyze.py` writes `results/analysis.json`: per-run QC (in-distribution accuracy, share of runs whose best epoch is the last one, which flags under-training), the unit table, P1–P3 with verdicts, secondary measures and the p dial. `expt_v1/report.py` prints the tables that go verbatim into `RESULTS.md`. Units within a split share a training run, so the unit bootstrap understates between-run variance; the split-level spread is reported alongside (disclosed limit, chosen before any data).

## 5. Verification (`expt_v1/verify.py`)

- `--stage remote` (Sherlock post job, on a GPU): prepared data hash to `data_sha256.json`; every run of every frozen run list present and no `.error` file; two replays from scratch (the first text cell and the first vision cell) agree with the originals within 0.03 in in-distribution and held-out accuracy; structure and analysis recomputed; `results/MANIFEST.sha256` written.
- `--stage local` (here, after the fetch): every file matches the manifest; `analysis.json` recomputed from the run files within 1e-9 (numpy and scipy versions differ between Sherlock and here); structure recomputed from local data within 1e-4; every verdict and the two p = 0 means in `RESULTS.md` match `analysis.json`.
- `expt_v1/test_splits.py`: maps are partitions with the right sizes, the real CIFAR-100 map equals the dataset's superclasses, the budget is exactly N at every p, no held-out class in validation, n_h = round(p × held-out items), validation and N do not depend on p or seed.

## 6. Compute

Sherlock only (Brando, 10-20-2026: nothing billed through Rylan; transformers only; V100s allowed). `launch/submit_exp22.sh` from `$SCRATCH/exp22`: a foreground `srun -p dev` builds the wheel-only venv `$SCRATCH/exp22_venv` (python 3.12.1, torch 2.6.0 with bundled CUDA 12.4, numpy 2.2.6, scipy 1.16.3, scikit-learn 1.7.2, pillow, pyarrow), downloads the raw data and checks every hash and the split tests; only then one GPU array per phase goes to `-p gpu` with `-C GPU_GEN:VLT` (V100-class; `GPU_CON=` allows any GPU if the Volta queue stalls, a deviation to be logged), one array task per run, Slurm mail END,FAIL to brandojazz@gmail.com and brando.science@gmail.com; a post job (after every array ends) runs `verify.py --stage remote` and writes `$SCRATCH/exp22/exp22_results.tgz`. Estimated cost: vision ≈ 10–20 GPU-minutes per run on a V100, text ≈ 3–5, about 25 GPU-hours in all. CPU was ruled out: the pilot's tiny depth-1 ViT trained at about 20 images/s per core.

## 7. Code-path pilot (CPU, this machine, seed 22909)

Run 10-20-2026 09:05–09:45 with `python run.py --pilot` (one thread, while Experiment 21 held this machine's CPU), then `structure.py`, `analyze.py`, `report.py` and `verify.py --pilot` on `results_pilot/` (not committed). Pilot sizes (`config.json` "pilot"): ViT depth 2 (0.92 M parameters), 2 epochs, 1,024 training and 256 validation images drawn from the split's full training and validation sets, 1,000 test images; text transformer depth 1, 64 tokens, 2 epochs, 400/100/400 documents; feature ViT depth 1, 1 epoch on 1,024 CIFAR-10 images, then features of all 60,000 CIFAR-100 images. Cells: the feature extractor; vision real map k=0 at p=0 and p=0.2; vision shuffled k=0 p=0; the same three for text.

What it showed: every run finished and wrote its result; both arms learn above chance even at this size (vision in-distribution test accuracy 0.12–0.13 real and 0.08 shuffled against 0.05 chance; text 0.39 against 0.25; CIFAR-10 feature ViT 0.20 against 0.10); the p dial puts exactly round(p × held-out items) into a split of the fixed budget (2,000 of 38,000 images at p=0.2 for vision; 446 of 6,345 documents for text); `structure.py` computes all 232 units (200 vision, 32 text; real-map centroid margin above shuffled-map margin in both arms, as built); `analyze.py` and `report.py` produce every table and verdict line; `verify.py --pilot` remote and local stages run end to end (replays at pilot size). CPU speed: the depth-1 ViT forward pass ran at about 37 images/s on one loaded core, which is why the grid runs on Sherlock GPUs.

Changes the pilot forced (code only, before the freeze): the 20 Newsgroups archive is read in archive order (a sorted walk re-read the gzip stream for every member); the analysis bootstrap is vectorized (the loop version took over 10 minutes); the pilot evaluates on test subsets. No design value changed. The pilot's accuracies are not evidence about P1–P3 and are not used.

## 8. Disclosures

- Experiments 11–21 and their results were seen before this design; Experiment 22 is the follow-up Brando asked for after reading them.
- The first draft of this design (10-20-2026 08:18) used a small CNN and TF-IDF + logistic regression / MLP models; Brando asked at 08:19 for transformers only, before any code or data of this experiment existed. Nothing else changed with the models.
- No Experiment 22 accuracy at real scale was seen before the freeze. The pilot's numbers come from 1–2 epoch, depth-1/2 models on ≤1,024 training items and are reported only to show the code runs.
- Measured structure of Section 2.5 is close in spirit to a nearest-centroid classifier on frozen features, so P2 is partly a validity check of the measure; P1 (the shuffled-map control) is the causal test of H_B.

## 9. Deviations

None yet.
