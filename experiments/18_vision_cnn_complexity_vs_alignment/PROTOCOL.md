# Protocol: vision replication of complexity versus alignment and of hard-to-easy transfer (`18_vision_cnn_complexity_vs_alignment`)

Status: FROZEN 10-19-2026 after the calibration pilots (Sections 6 and 6.1; the changes they forced are marked there and in the sections they touch). Original DRAFT design 10-18-2026, written before any run. The calibration pilot (Section 6) may change only the values Section 6 names; its outcome is written into Section 6 and frozen in its own commit before any gated run. After the freeze nothing in Sections 2–5 changes.

## 1. Why

Experiment 13-small (GINC-style HMM mixtures, CPU) found that complexity is real (G1) but that alignment does not dominate it: the elasticity ratio $E = e_\alpha/e_K$ was 0.80 [0.79, 0.83], below the 1 the alignment claim needs. That result comes from one synthetic sequence family and a partly trained model. This experiment asks the same two questions in a second modality, with small convolutional networks on the MNIST family and CIFAR-10, trained on CPU:

1. **Elasticity direction.** Does the vision setting give the same direction as Experiment 13-small ($E < 1$: a log-unit of complexity costs more data than a log-unit of alignment), or the opposite?
2. **Hard to easy (Brando's Putnam vs GSM8K question).** Does training only on hard tasks carry the easy ones, at matched compute? If difficulty were a total order, hard-only training would do at least as well on easy targets as easy-only training.

## 2. Design

### 2.1 Inputs and datasets

A training or test example is a **pair image**: two images of the same dataset side by side, with class indices $a$ (left) and $b$ (right). Primary datasets: MNIST, Fashion-MNIST, KMNIST (28×28 grey). CIFAR-10 (32×32 colour) is a full arm with the same grids and seeds. Training pairs draw both images uniformly from the dataset's train split; evaluation pairs draw from its test split only, so no evaluation image is a training image. Raw files and their SHA-256 are recorded by `expt_v1/prep.py`.

### 2.2 Tasks and their exact complexity

A task is a label function $f(a, b)$ of the two class indices (`expt_v1/tasks.py`). The network processes the two images with one shared tower (Section 2.6), so the information it must extract is per image. **Exact complexity** $K$ (bits) is the sum over the two slots of $\log_2$ of the number of classes of that slot's minimal sufficient partition: $d \sim d'$ in the left slot iff $f(d, b) = f(d', b)$ for every $b$, and likewise for the right slot. It is computed by brute force on the 10×10 table and unit-tested against hand-derived values. Class indices are used as labels in every dataset (for Fashion-MNIST and CIFAR-10 the arithmetic over class indices has no visual meaning; it is the same abstract task).

Main grid: 4 levels × 2 targets.

| Level | $K$ (bits) | Targets |
|---|---|---|
| L1 | 1 | parity of $a$; $[b \ge 5]$ |
| L2 | 2 | parity of $a+b$; $[a\ge5] \oplus [b\ge5]$ |
| L3 | 3.32 | identity of $a$; identity of $b$ |
| L4 | 6.64 | $a+b$ (19 classes); $a-b$ (19 classes) — *changed in Section 6.1; the pre-pilot targets were $(a+b) \bmod 10$ and $(a-b) \bmod 10$* |

**Primary outcome** $\varepsilon$ = test error divided by the task's chance error (the error of the best constant guess under uniform independent $a, b$: 0.5 for the binary L1/L2 targets, 0.9 for the 10-class ones). $\varepsilon = 1$ is chance and $\varepsilon = 0$ perfect, so levels with different numbers of classes are on one scale. Raw error and cross-entropy (bits) are also stored. There is no exact oracle floor here (unlike Experiment 13): the laws are fit without an offset (`ident_bits = 0`), and the image-level Bayes error is a disclosed limit.

### 2.3 Alignment construction

$\alpha$ = the share of training pairs whose images come from the **test (aligned) distribution**. The remaining $1-\alpha$ carry the same task labels but images passed through a **mismatch** $M$. Example $k$ (0-based) of the stream is aligned iff $\lfloor (k+1)\alpha \rfloor - \lfloor k\alpha \rfloor = 1$ (exact rational arithmetic), so after $m$ examples exactly $\lfloor m\alpha\rfloor$ are aligned, at every checkpoint. Grid: $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$.

- **Primary mismatch** (every dataset): rotation by 90°.
- **Mismatch-type sweep** (MNIST only): scale 0.5 (2× downsample, centred), thick strokes (3×3 grey dilation), thin strokes (2×2 grey erosion), Fashion-MNIST images of the same class index, KMNIST images of the same class index. Rotation is the shared reference. The two swaps are the vision analogue of disjoint support: the misaligned data teach the label rule but none of the target's visual features.

This is the alignment-coefficient notion (share of training data that matches the evaluation distribution). Experiment 13's distractors were other concepts on the same tokens; here they are the same concept seen through a shifted input. The difference is disclosed in the results.

**Common random numbers.** Within a (dataset, seed), the $k$-th training pair uses the same two train-image indices (and, for swaps, the same uniform draw for the swapped image of each class; for CIFAR-10, the same crop and flip) in every run, across all targets, $\alpha$ levels and mismatches. Model initialization is identical across every run of a (dataset, seed); each head is initialized from (seed, task) alone.

### 2.4 Hard-to-easy arm (H)

Multi-task training over the aligned distribution, one linear head per task, shared tower and trunk. Non-target examples cycle through the pool's tasks in order.

| Pool | Tasks (all 4 at matched compute) |
|---|---|
| easy-only | parity of $a$, $[b\ge5]$, parity of $a+b$, $[a\ge5]\oplus[b\ge5]$ (1–2 bits) |
| hard-only | $a+b$, $a-b$, $a+3b$, $3a+b$ (6.64 bits each; Section 6.1) |
| mixed | parity of $a$, $[a\ge5]\oplus[b\ge5]$, $a+b$, $a+3b$ |
| hard-mod (secondary, descriptive) | $(a+b)$, $(a-b)$, $(a+3b)$, $(3a+b)$, all mod 10 (6.64 bits): the pre-pilot hard pool |

Held-out targets, none in any pool: **easy** $[a \bmod 3 = 0]$ (1 bit), $[a\ge7]\oplus[b\le2]$ (2 bits); **hard** $a+7b$ (6.64 bits, same family as the hard pool; Section 6.1 replaced $(a+7b+2) \bmod 10$), $[a<b]$ (6.64 bits with a binary output). One extra anchor per dataset and seed, $(a+b) \bmod 10$ at $\alpha=1$, records when the mod-10 plateau ends (descriptive).

Three transfer readouts, all at matched compute (same model, same number of training pairs):

1. **Joint, $\alpha = 0.05$ (primary).** The target is 5% of the stream (same rule as Section 2.3), the pool the other 95%. This matches Experiment 16/17's $\alpha = 0.05$ arm.
2. **Linear probe (secondary).** Pool-only training; at each checkpoint, a multinomial logistic regression on the frozen 128-unit trunk features, fit on 1,000 aligned train pairs (per seed) with L2 penalty $10^{-3}$ (LBFGS, 200 iterations, standardized features), scored on the evaluation pairs. Also at $T = 0$ (random features).
3. **Anchor.** Target only, $\alpha = 1$: the target's own learning curve.

### 2.5 Evaluation pairs

4,096 fixed pairs from the test split (seed `20261018 + 7`), the same index pairs for every run of a dataset. Standard error of the raw error is at most $\sqrt{0.25/4096} = 0.008$. `--check-only` records how many test images are pixel-identical to some train image (a property of the public datasets, reported, not filtered).

### 2.6 Model, optimizer, checkpoints

- **Tower** shared by the two images: conv 5×5 stride 2 ($w$ channels), conv 3×3 stride 2 ($2w$), conv 3×3 stride 2 ($4w$), ReLU after each, 2×2 adaptive average pool, linear to 64, ReLU. **Trunk:** the two 64-d vectors concatenated, linear 128, ReLU. **Heads:** linear per task. $w = 16$ for the MNIST family (57,802 parameters with one 10-class head), $w = 32$ for CIFAR-10 (145,418). No normalization layers, no dropout. CIFAR-10 uses random crops (4-pixel padding) and horizontal flips from the per-seed draws; the MNIST family uses no augmentation.
- **Optimizer:** AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1 on weight matrices only, gradient clipping 1.0, 32 warmup steps, then a constant learning rate $10^{-3}$ (as in Experiment 13: every checkpoint is a point on one run's curve). Batch 64 pairs. Float32, one thread per run, deterministic algorithms.
- **Checkpoints** ($T$ = training pairs seen): MNIST family $2^{10}, \dots, 2^{20}$; CIFAR-10 $2^{12}, \dots, 2^{21}$ (Section 6.1; pre-pilot $2^{17}$ and $2^{19}$). $2^{20}$ pairs are 2.1 M image views, 35 epochs of MNIST; $2^{21}$ CIFAR-10 pairs are 84 epochs.

### 2.7 Seeds and grids

Seeds `18101, 18202, 18303, 18404`; pilot seed `18909` (never reused); analysis seed `29101826`; 20,000 bootstrap draws.

| Grid | Runs |
|---|---|
| Main, per dataset: 8 targets × 4 $\alpha$ × 4 seeds | 128 × 4 datasets = 512 |
| Mismatch sweep, MNIST: 5 types × 8 targets × $\alpha \in \{0.05, 0.4\}$ × seeds 18101, 18202 ($\alpha = 1$ shared with the main grid) | 160 |
| H, per dataset and seed: 4 pool-only + 4 pools × 4 targets joint + 5 anchors = 25 | 100 × 4 datasets = 400 |
| **Total** | **1,072** |

Held-out cells (main grid): level L3, $\alpha = 0.15$, and the largest checkpoint. A row is a fit row iff none of its coordinates is held out.

## 3. Measurements

Per run and checkpoint: raw error, cross-entropy (bits) and $\varepsilon$ on the evaluation pairs for every head; for $\alpha < 1$, the error on 1,024 misaligned-source pairs (training-distribution error, not gated); for H pool-only and joint runs, probe error on the held-out targets; wall time and training pairs per second. The final weights of every run are saved (outside git) so `verify.py` can recompute the last checkpoint's evaluation.

## 4. Analysis (per dataset; MNIST is primary)

- **Laws** (Experiment 13's `laws.py`, unchanged, offset off): `M0_scale`, `M_align_eff` $A(\alpha^\eta \tilde T)^{-\gamma}$, `M_K_coef`, `M_K_exp`, `M_K_both` $A\tilde K^\kappa(\alpha^\eta\tilde T)^{-\gamma(K)}$, with $K$ in bits, $K_{\text{ref}}$ the geometric mean of the fit levels' $K$, $T_0$ the smallest checkpoint. Fit per seed on fit rows; same bounds, starts and 64-start grid as Experiment 13. **Learning-regime filter:** rows with $\varepsilon > 0.8$ (still within 20% of chance) are dropped from every fit, because a floor-less power law cannot represent the plateau at chance; their count is reported. `independent_fit.py` refits `M_K_both` with L-BFGS-B and must agree on held-out RMSE to $10^{-3}$ relative.
- **Model-free $T^\star$:** first crossing of $\varepsilon^\star$, log-log interpolated between checkpoints (Experiment 13's `mf_tstar`); censored cells are set to the nearest checkpoint and counted.
- **$\varepsilon^\star$ per dataset** (Section 6.1): $\max(0.10, 1.5\,\bar\varepsilon_{L4})$ rounded up to 0.01 and capped at 0.60, where $\bar\varepsilon_{L4}$ is the mean $\varepsilon$ of the two L4 targets at $\alpha = 1$ at the largest fit checkpoint (all seeds). It uses $\alpha = 1$ cells only and is computed by `analyze.py`, not chosen by hand.
- **Elasticities** at $\varepsilon^\star$: $e_\alpha = \eta$, $e_K = \partial \ln T^\star/\partial \ln K$ at $K_{\text{ref}}$ from `M_K_both` refit on all rows (descriptive); $E = \bar e_\alpha/\bar e_K$. Model-free $E_{\text{mf}}$: per seed, least squares of $\log_2 T^\star$ on $\log_2(1/\alpha)$ and $\log_2 K$ over uncensored cells, ratio of the two coefficients, averaged over seeds; evaluated at $2\varepsilon^\star$ if more than half of the cells are censored in the median seed. A secondary $E$ uses $K$ as the number of per-slot classes multiplied, $2^{K_{\text{bits}}}$.
- **Uncertainty:** 95% percentile intervals from a seed-block bootstrap (20,000 draws). With 4 seeds there are only 35 distinct resamples; a second, two-stage bootstrap (seeds, then the 2 targets within each level) is reported.
- **Mismatch sweep:** per mismatch type, `M_K_both` fit per seed on all rows of that type ($\alpha \in \{0.05, 0.4\}$ plus the shared $\alpha = 1$ rows of the same seeds); $\eta$, $E$ and the model-free $M_\alpha = T^\star(0.05)/T^\star(1)$ per level. Descriptive (2 seeds).
- **H arm:** for each condition, the learning-curve score $S = $ mean over checkpoints of $\log_2 \max(\varepsilon, \varepsilon_{\text{res}})$, $\varepsilon_{\text{res}} = 0.005$ (log area under the learning curve, so saturated end points do not dominate). Ratio $\rho_{\text{HE}} = 2^{\bar S_{\text{hard-only}} - \bar S_{\text{easy-only}}}$ on the two easy targets (mean over targets, then seeds); $\rho_{\text{EH}} = 2^{\bar S_{\text{easy-only}} - \bar S_{\text{hard-only}}}$ on the two hard targets. Also reported at the largest checkpoint, for mixed against both, and against the anchor at matched target examples. The hard-mod pool gets the same ratios against easy-only and against hard-only (descriptive, no gate).

## 5. Gates (per dataset; every gate ends with its verdict, reported with equal prominence)

- **G1, complexity is real.** $M_K$ at $\alpha = 1$ from model-free $T^\star$, geometric mean over the two L4 targets over the two L1 targets. `SUPPORTED` iff $M_K \ge 2.0$ and the seed-block and two-stage 95% lower bounds are $> 1$.
- **G2, complexity adds held-out value.** The $K$ law (`M_K_coef` or `M_K_exp`, whichever has lower pooled fit-row RMSE) against `M_align_eff` on held-out rows: `SUPPORTED` iff relative RMSE reduction $\ge 10\%$ with 95% lower bound $> 0$, lower RMSE on the held-out level's rows alone, and 0 invalid forecasts.
- **G4, alignment dominates.** `SUPPORTED` iff $E > 1$ with seed-block 95% lower bound $> 1$, the secondary $E > 1$, and $E_{\text{mf}} > 1$.
- **Replication of Experiment 13-small's direction (headline).** `SAME_DIRECTION` iff $E < 1$ with 95% upper bound $< 1$ and $E_{\text{mf}} < 1$; `OPPOSITE` iff G4 is `SUPPORTED`; otherwise `UNRESOLVED`.
- **GH1, hard-only carries the easy end** (joint $\alpha = 0.05$ primary; probe reported with the same rule). `CARRIES` iff the 95% upper bound of $\rho_{\text{HE}}$ is $< 1.10$ (non-inferior within 10%); `DOES_NOT_CARRY` iff its 95% lower bound is $> 1.10$; otherwise `UNRESOLVED`.
- **GH2, easy-only does not carry the hard end.** `SUPPORTED` iff $\rho_{\text{EH}} > 1.10$ with 95% lower bound $> 1.10$.

G3 (coefficient or exponent) is computed as in Experiment 13 and reported, not gated. Experiment 13's G5 (position split) has no analogue here.

### 5.1 Predictions, written before any run (10-18-2026)

- **G1:** `SUPPORTED` in every dataset. The L4 targets need both digits identified; I expect $M_K$ of 3–10× on the MNIST family.
- **Headline direction:** `SAME_DIRECTION` on the MNIST family, with $E$ between 0.4 and 0.9: rotated digits still teach the label rule and part of the stroke features, so alignment's elasticity $\eta$ stays below 1, while complexity costs more than one log-unit per log-unit of $K$. For the two swap mismatches I expect $\eta$ to be the largest of the sweep (the misaligned data teach no target features), so $E$ is closest to or above 1 there. CIFAR-10: same direction, lower confidence.
- **GH1:** `CARRIES` on the MNIST family, against the sequence intuition. Every hard task here forces the network to identify both class indices completely, and every easy target is a function of those indices, so hard-only features should contain what the easy targets need, while easy-pool features (parities, thresholds) do not determine $a \bmod 3$. If GH1 instead comes out `DOES_NOT_CARRY`, difficulty is not a ladder even when the hard tasks are informationally sufficient, which would be the stronger and more surprising result.
- **GH2:** `SUPPORTED` (easy-only does not carry the hard end).

These are the running agent's forecasts, not gates; the gates above decide.

**Amended predictions after the pilots, before any gated run (10-19-2026).** The pilots (Section 6.1) changed the L4 targets and the hard pool, so the forecasts that depend on them are restated here: G1 `SUPPORTED` on the MNIST family ($M_K$ 3–10×) and probably on CIFAR-10; headline `SAME_DIRECTION` on the MNIST family, $E$ 0.4–0.9, unchanged; GH1 `CARRIES` with the additive hard pool (it must identify both digits, which determines every easy target); GH2 `SUPPORTED`. Hard-mod pool (descriptive): it will not leave the plateau in most seeds within $2^{20}$ pairs, and its features will carry the easy targets worse than random initialization, as in pilot 2.

## 6. Calibration pilot (disclosed, before the freeze)

Seed `18909`: MNIST, targets parity of $a$ and $(a+b) \bmod 10$, $\alpha \in \{0.05, 0.4\}$; Fashion-MNIST, KMNIST and CIFAR-10, $(a+b) \bmod 10$ at $\alpha = 0.4$; one MNIST hard-pool H run. Rules, fixed now:

- **$\varepsilon^\star$** (default 0.10 for every dataset): if $(a+b) \bmod 10$ at $\alpha = 0.4$ has $\varepsilon > 2\varepsilon^\star$ at the largest fit checkpoint, that dataset's $\varepsilon^\star$ is raised to its $\varepsilon$ there, rounded up to the next 0.01.
- **CIFAR-10 length:** the pilot run is extended to $2^{20}$. If its $\varepsilon$ at $2^{20}$ is more than 10% below its $\varepsilon$ at $2^{19}$, CIFAR-10 checkpoints become $2^{13}, \dots, 2^{20}$; otherwise they stay $2^{12}, \dots, 2^{19}$.
- **Throughput:** measured pairs per second set the wall-time estimate; runs that cannot finish on the local 4-core box go to Sherlock's CPU partition with the same code (`launch/`).

**Pilot outcome (seed 18909, 10-19-2026, local CPU).** $\varepsilon$ of $(a+b) \bmod 10$ at $\alpha = 0.4$ at the largest fit checkpoint ($2^{16}$): MNIST 1.002, Fashion-MNIST 0.887, KMNIST 1.000; at $2^{17}$ 0.990, 0.779, 0.997. MNIST parity of $a$ reached 0.056 ($\alpha=0.4$) and 0.210 ($\alpha=0.05$) at $2^{17}$. The MNIST hard pool (four mod-10 targets) stayed at chance on every head. CIFAR-10 $(a+b) \bmod 10$ at $\alpha=0.4$: 0.927 at $2^{19}$, 0.887 at $2^{20}$ (4.3% lower, so the CIFAR length rule as written keeps $2^{19}$). Throughput on one core: 850 pairs/s (MNIST family, 4 runs in parallel), 430 pairs/s (CIFAR-10). Read literally, the $\varepsilon^\star$ rule would set $\varepsilon^\star \approx 0.99$ for MNIST and KMNIST, and GH1 would be decided by a hard pool that never learned. Both outcomes measure the plateau, not the questions, so Section 6.1 changes the design before the freeze.

### 6.1 Second pilot and the changes it forced (disclosed deviation, before the freeze)

The mod-10 targets are XOR-like: under uniform $a, b$, the label is independent of either digit alone, so a network gets no first-order gradient signal until it has partly identified both digits, and training sits at chance for a variable time. Pilot 2 (seed 18909, MNIST, to $2^{20}$, `expt_v1/pilot2.py`, results in `results/pilot2/`): $(a+b) \bmod 10$ at $\alpha = 1$ stays at chance through $2^{15}$, is at 0.70 at $2^{17}$ and leaves the plateau between $2^{17}$ and $2^{18}$ (0.096), reaching 0.031 at $2^{20}$; $a+b$ (19 classes, the same $K = 6.64$ bits, each digit alone informative) is at 0.29 at $2^{16}$, 0.081 at $2^{17}$, 0.029 at $2^{20}$; the four-target mod-10 hard pool stays at chance on every head through $2^{20}$, and its trunk features get worse than random initialization for the held-out probes (probe $\varepsilon$ on $[a<b]$ 0.67 at $T=0$, 0.99 from $2^{18}$).

Changes, fixed now, before any gated run:

1. **L4 targets** become $a+b$ and $a-b$ (19 classes each, $K = 6.64$ bits, unchanged). **Hard pool** becomes $a+b$, $a-b$, $a+3b$, $3a+b$ (non-modular, $K = 6.64$ each); the mixed pool uses $a+b$ and $a+3b$; the held-out hard target $(a+7b+2) \bmod 10$ becomes $a+7b$. The pre-pilot mod-10 hard pool is kept as a fourth, secondary pool (hard-mod), and $(a+b) \bmod 10$ at $\alpha=1$ as an extra anchor, both descriptive: same $K$, very different learnability, which bears on the difficulty-confound question directly.
2. **Training length**: MNIST family to $2^{20}$ pairs, CIFAR-10 to $2^{21}$ (Brando asked for runs trained to convergence rather than read mid-training, 10-18-2026; at $2^{17}$ even L1 at $\alpha=0.05$ was still at 0.21).
3. **$\varepsilon^\star$** is set per dataset by the rule in Section 4 from the run's own $\alpha=1$ L4 cells, replacing the pilot rule, which no longer applies (its target was replaced). The model-free fallback to $2\varepsilon^\star$ is unchanged.
4. **Where it runs**: the cost is now about 700 core-hours (MNIST family about 310, CIFAR-10 about 390), too much for the shared 4-core box, so every phase runs on Sherlock's `normal` CPU partition through `launch/` (Section 7).
5. **Verification across machines** (Section 8): runs are compared exactly on the CPU model they were trained on, with a cross-CPU tolerance elsewhere.

## 7. Compute and execution

Every phase of the grid runs on Sherlock's `normal` CPU partition (Section 6.1), submitted once by `launch/submit_exp18.sh` from a package built at the freeze commit: a prep job builds the environment, downloads the raw files and checks their hashes and the decoded arrays against `expt_v1/data_sha256.json`; one Slurm array per phase (one CPU and one run per task) runs after it; a post job runs `verify.py --stage remote` and packs `results/` for transfer. Results are the same files as local runs. If Sherlock is unavailable, the same phases run locally under the shared box's CPU lock (`scratch/at343/CPU_LOCK_AT-439`), one phase at a time, in the order MNIST main, MNIST H, Fashion-MNIST and KMNIST main, their H arms, the mismatch sweep, CIFAR-10 main, CIFAR-10 H.

## 8. Verification (`verify.py`)

*Amended in Section 6.1:* data are checked by the SHA-256 of the decoded arrays (npz files rebuilt from the same raw files on another machine differ only in zip timestamps); the exact (1e-12) recomputation and the retraining run on Sherlock as `verify.py --stage remote` inside the post job, on runs trained on the same CPU model, and the local `verify.py` recomputes every run within a cross-CPU tolerance of 3 of 4,096 evaluation pairs (1e-4 bits on cross-entropy) and requires the remote stage to have passed.

Every run present and complete; aligned and target example counts equal $\lfloor T\alpha \rfloor$ at every checkpoint; every task's $K$ recomputed; the last checkpoint's evaluation recomputed from the saved weights to $10^{-12}$; a sample of runs retrained end to end and their rows compared exactly (determinism); `analyze.py` rerun into a scratch copy with identical numbers and verdicts; the independent fitter agrees; every number in the results file is read from `analysis.json`.

## 9. Limits

No exact oracle floor; error rates on 4,096 pairs; four seeds (35 distinct bootstrap resamples); small CNNs at CPU scale; label arithmetic over class indices is abstract for Fashion-MNIST and CIFAR-10; one primary mismatch type for the elasticity grid.
