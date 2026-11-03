# Protocol: Experiment 24, does the similarity of the non-target data decide whether alignment or complexity dominates? (`24_similarity_alignment_complexity`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/24_similarity_alignment_complexity/PROTOCOL.md>

Owner: Laolu Oguneye. Written 11-03-2026, before any run of this experiment. Experiments 11–23, the 13-GPU results (PR #28) and the 13-GPU censoring reanalysis (`13_ginc_hmm_complexity_vs_alignment/AT1346_censoring_reanalysis/`) are disclosed as seen.

## 1. Why this experiment

The complexity-versus-alignment series asks what makes a target expensive to learn: its complexity $K$, or how little of the training data is about it (the target share $\alpha$). The answers disagree.

- With synthetic hidden Markov model (HMM) languages, the fitted laws give $E = e_\alpha / e_K < 1$: complexity moves the data bill more than alignment does (13-small 0.80, Experiment 23 0.67, 13-GPU 0.58). A censoring-aware reading of 13-GPU's crossing times gives $E \approx 1$ (AT1346).
- In vision (Experiment 18) $E$ splits by dataset (MNIST 1.18, KMNIST 1.43, Fashion-MNIST 0.36, CIFAR-10 0.78). Its sweep over the kind of non-target data is the clue: with similar non-target images (thicker or thinner strokes) $\eta = e_\alpha$ is about 0.2 and $E \approx 0.3$; with dissimilar ones (rotation, a different dataset) $\eta$ is 0.66–0.96 and $E$ is 0.89–1.17.
- In 13-GPU the non-target concepts come from the same generator and vocabulary as the target, and for the 85M model 20 times less target data cost only about 1.3 times more training. Related data substitutes for target data, more so in bigger models, as in transfer scaling laws ([Hernandez et al. 2021](https://arxiv.org/abs/2102.01293): data transferred grows as $N^{0.38}$, more for closer distributions).
- "Alignment" has two meanings. The target share $\alpha$ is what these toys vary. The similarity of the training data to the target is what the Task2Vec alignment coefficient measures ([Chawla, Miranda, Koyejo et al.](https://arxiv.org/abs/2501.08496)). In Experiment 13, a target's complexity and its vocabulary overlap with the distractors moved together (correlation 0.99 across targets), so the two could not be separated.

**Hypothesis.** How much alignment matters depends on how similar the non-target data is to the target. When the rest of the data shares the target's structure, the exact target share matters little ($E < 1$). When it shares nothing, the share is all that matters ($E > 1$). Bigger models exploit similar data more, so they lower $e_\alpha$ only where similarity exists.

## 2. Design

### 2.1 Concepts (the language of Experiment 13, unchanged)

A concept is a GINC-style sparse HMM in Experiment 13's language $\mathcal L_{\text{HMM}}$ (its PROTOCOL Sections 2.1–2.3): $n$ hidden states, 3 successor states and 4 emission tokens per state, probabilities in multiples of $2^{-6}$, a uniform start state, minimal state count certified exactly by process rank, and $K_{\text{bits}}$ its encoding length. Targets come from Experiment 13's acceptance procedure (`run.build_concepts`, unchanged code) on a new stream (task-sampling seed `20261013`, arm index 2). There are 3 targets per level $n \in \{8, 32, 64\}$ (9 targets) and 1 pilot concept per level. The entropy band is calibrated on each level's first 64 rank-certified candidates, at ±0.15 bits around their pooled median $h_{\text{ref}}$ (widened to ±0.30 only if a level cannot fill), as in Experiment 13. Vocabulary: 150 tokens for targets.

### 2.2 Distractor pools: the similarity dial

Every target (and the $n = 32$ pilot concept, family level only) gets its own pool of $D = 4$ distractors at each of four similarity levels. All distractors have the target's $n$. Each one comes from its own PCG64 stream seeded (`20261013`, 24, target id, level index, $d$) and passes the targets' checks: strongly connected and aperiodic, process rank $= n$ (certified), and quick entropy rate within the targets' band of $h_{\text{ref}}$. So every concept in the experiment has the same noise level.

| Level | Construction | Shared structure with the target |
|---|---|---|
| `copy20` | the target with round(0.2 n) states (at least 1) re-drawn: their successor set, transition numerators, emission tokens and emission numerators taken from a fresh candidate | ≈ 80% of state rows identical |
| `copy50` | the same with round(0.5 n) states re-drawn | ≈ 50% |
| `family` | an independent candidate of the same $n$, same 150 tokens (Experiment 13's kind of distractor) | 0 rows; vocabulary overlap grows with $n$ |
| `disjoint` | an independent candidate of the same $n$ whose emission tokens are shifted to 150–299 | nothing: 0 rows, 0 tokens |

Every model sees a 300-token vocabulary plus a beginning-of-sequence token (vocabulary 301) in every run, so the architecture, the initialization and the non-embedding parameter count $N$ are identical across similarity levels. **Disclosed at the design stage:** within `copy` and `family` pools, vocabulary overlap with the target still rises with $n$ (bigger machines use more of the 150 tokens), so the controlled dial is shared structure, and only `disjoint` removes vocabulary overlap. Identification stays easy at every level: even with `copy20` pools the exact posterior on the true target after 16 tokens averages 0.87 at α = 0.05 and reaches 1 by the end of a document, so the identification floor $\varepsilon_{\text{id}}$ is about $\log_2(1/\alpha)/128$ at every level.

### 2.3 Alignment, documents and the grid

- $\alpha \in \{0.01, 0.05, 0.2, 0.5, 1\}$ (a 100-fold range), with Experiment 13's exact rule: document $k$ is a target document iff $\lfloor (k+1)\alpha \rfloor - \lfloor k\alpha \rfloor = 1$. Non-target slots cycle through the 4 pool members in a seeded permutation. At $\alpha = 1$ there are no distractors, so similarity does not apply: one cell per target.
- Documents: 128 tokens from one concept, regenerated from (seed, concept, index). Common random numbers within a seed.
- Grid: 2 model sizes × 9 targets × 17 cells (α = 1, plus 4 similarity levels × 4 α < 1) × 3 seeds (`24101, 24202, 24303`) = **918 runs**, one GPU job (shard) each. Shard id = ((i_size · 9 + i_target) · 17 + i_cell) · 3 + i_seed.

### 2.4 Models, training and evaluation

- Models: Experiment 13-GPU's S (4 layers, d 256, 3,159,552 non-embedding parameters) and M (8 layers, d 512, 25,220,096), with vocabulary 301 (3,269,376 and 25,439,744 parameters in total). L is not run: 13-GPU showed the scale trend between S and M already, and this question does not need L.
- Training: exactly 13-GPU's code (`gpu_arm.py`). AdamW (β = 0.9, 0.95), weight decay 0.1 on matrices, gradient clipping 1.0, 200 warmup steps, then a constant learning rate, batches of 128 documents (16,384 tokens), bfloat16 autocast, and the same initialization for every run of one seed and size.
- **Every run trains to the cap, $T = 2^{28}$ tokens** (16,384 steps), with checkpoints at every doubling from $2^{20}$ to $2^{28}$ (9 checkpoints). There is no early stopping. Experiment 23's convergence rule is reported per cell, not used to stop. In 13-GPU, every same-family run at α ≥ 0.05 for S and M had reached 0.1 bits by $2^{28}$.
- Evaluation at every checkpoint, in float64 with deterministic algorithms on the same GPU: 1,024 held-out documents per target. Measures: excess ε over the exact oracle, the identification floor $\varepsilon_{\text{id}}$ (the exact Bayes mixture of target and pool under the training prior α, (1 − α)/4 each), and the learning term $\varepsilon_{\text{learn}} = \varepsilon - \varepsilon_{\text{id}}$, per slice (positions 1–16, 17–64, 65–128, all) and as the 128-position curve. Plus 64 documents per pool member (distractor loss, float32, reported).

### 2.5 Pilot (learning rate), disclosed after it runs

Seed `24919`, the $n = 32$ pilot concept with its own family pool, α = 0.2, $T$ to $2^{26}$. Learning rate is 1×, 2× or 4× the 13-GPU table value (S 1e-3, M 6e-4), so the grid extends above 13-GPU's choice (2×, the top of its grid). 6 runs. **Rule:** per size, the multiplier with the lowest pilot excess (all positions) at $2^{26}$. 2× wins if it is within 0.002 bits of the best. A nonfinite run never wins. The outcome goes into `config.json` `lr_selected` and a "Pilot disclosure" paragraph in Section 8, in one commit made before any shard. That commit changes nothing else. Seed `24919` is never used again.

## 3. Measurements

Per run and checkpoint, the rows above; per run, the training-stream SHA-256, per-checkpoint document counts, hardware and timing. **The data bill** $T^\star$ is the first crossing of $\varepsilon^\star = 0.1$ bits/token by $\varepsilon_{\text{learn}}$ (all positions), log-log interpolated between checkpoints. A run that never crosses by $2^{28}$ is right-censored ($T^\star \ge 2^{28}$). A run already below at $2^{20}$ is left-censored.

## 4. Analysis

### 4.1 Primary estimator: censoring-aware elasticities

Per similarity level $s$, a Tobit (censored normal) regression of $\ln T^\star$ on the runs of that level plus the shared α = 1 runs:

$$\ln T^\star = \mu_{\text{size}} - e_\alpha(s)\,\ln \alpha + e_K(s)\,\ln n + \sigma\,\epsilon ,$$

pooled over sizes (size intercepts, common slopes) and separately per size. Censored runs stay in through their censored likelihood: never dropped, never set to a checkpoint. $E(s) = e_\alpha(s)/e_K(s)$, with $K$ in states (in bits reported). Fitted by maximum likelihood (BFGS from the least-squares start). An independent fit (L-BFGS-B from zero) must agree to $10^{-4}$ relative on every coefficient.

### 4.2 Uncertainty

95% percentile intervals from a two-stage bootstrap: resample the 3 seeds, then the 3 targets within each level, both with replacement, 20,000 draws, analysis seed `24110326`. Each draw refits every regression with multiplicity weights. A degenerate draw is counted and reported, never dropped silently. **Every-seed rule:** each gate's point condition must also hold when the analysis is run on each seed alone.

### 4.3 G3's held-out comparison

The `copy50` level is held out. Two Tobit laws are fit on the other three levels plus α = 1 (sizes pooled):
- α-only: $\ln T^\star = \mu_{\text{size}} + b\ln\alpha + c\ln n$;
- similarity: adds $f\,\ln\alpha$ and $\mathbb 1_{\text{disjoint}}\ln\alpha$, with $f$ = the nominal shared fraction (0.8, 0.5, 0, 0).

They are scored on the held-out `copy50` runs by a censored residual: the plain error for crossed runs, and for a censored run only the error on the wrong side of its bound. The score is the root-mean-square of these residuals.

### 4.4 Gates (pooled over sizes unless stated; every gate also needs the every-seed rule)

| Gate | Claim | SUPPORTED iff |
|---|---|---|
| **G1** | Similarity sets alignment's price | $e_\alpha$ rises monotonically `copy20` < `copy50` < `family` < `disjoint` (point estimates), and $\Delta = e_\alpha(\text{disjoint}) - e_\alpha(\text{copy20}) > 0$ with 95% lower bound > 0 |
| **G2** | $E$ crosses 1 | $E(\text{disjoint}) > 1$ with lower bound > 1 **and** $E(\text{copy20}) < 1$ with upper bound < 1. Named one-sided outcomes: "E > 1 at every similarity" and "E < 1 at every similarity" |
| **G3** | Similarity adds predictive value | the similarity law cuts the held-out `copy50` error by ≥ 10% with lower bound > 0 (every seed: reduction > 0) |
| **G4** | Complexity is real at every similarity | $e_K(s) > 0$ with lower bound > 0 for all four levels |
| **G5** | Bigger models use similar data | with $\delta(s) = e_\alpha^{S}(s) - e_\alpha^{M}(s)$: $A = \text{mean}\,\delta(\text{copy20, copy50, family}) > 0$ with lower bound > 0, **and** $A - \delta(\text{disjoint}) > 0$ with lower bound > 0 |

### 4.5 Reported, not gated

- The same quantities at $\varepsilon^\star = 0.05$ and 0.2, with censoring counts per (size, level, α).
- $K$ in bits.
- **The fitted-law $E$** (secondary): Experiment 13's `M_K_both` law (identification-floor offset, complexity-dependent exponent) refit per size and level on all rows, with its elasticities at 0.1 bits and $K_{\text{ref}}$.
- Experiment 23's convergence rule per cell, between $2^{26}$ and $2^{28}$.
- $\varepsilon_{\text{id}}$ and the posterior after 16 tokens per level and α; the loss on each run's own distractors.
- **The Task2Vec alignment coefficient** of every training mix with its target (`task2vec24.py`), with its Spearman correlation against $\ln T^\star$ (crossed runs) and its means by level and α. Probe: an S model (seed `24999`) trained on a uniform mixture of all 160 concepts for $2^{24}$ tokens at the selected S learning rate. Embedding of a batch of 64 documents: the L2-normalized diagonal empirical Fisher information of the probe's last block and final LayerNorm. Coefficient: 1 − the mean cosine distance over 4 × 4 batch pairs (mix documents 0–255 of seed 24101's stream; target held-out documents 0–255). Never a gate.

### 4.6 What each outcome would mean

- **G1 and G2 SUPPORTED:** the series' disagreement is explained. Alignment matters in proportion to how little the rest of the data shares with the target. The toy $E < 1$ and the alignment-coefficient view are two ends of one dial.
- **G1 SUPPORTED, G2 not:** similarity moves alignment's price, but not enough to flip which dial dominates in this range.
- **G1 NOT_ESTABLISHED:** the share of exact target data matters about equally whatever the rest of the data is. The similarity account of Experiment 18's sweep does not carry to HMM languages.
- **G5 SUPPORTED:** scale buys transfer from similar data specifically. 13-GPU's falling $\eta$ is a transfer effect.
- **G4 NOT_ESTABLISHED at some level:** complexity's cost depends on what else is in the data.

## 5. Predictions

Laolu records his predictions for G1–G5 (verdict plus a number each) in `CKPT_exp24.md`, in a commit made after the shards start and before any analysis runs.

## 6. Verification (`verify24.py`)

- **Per shard, on its GPU right after training:**
  - every file in DONE matches its SHA-256;
  - the shard ran the current sources and concepts;
  - the training stream regenerates (SHA-256 and document counts);
  - all 9 checkpoints replay in float64 to < $10^{-9}$ bits (per slice and per position).

  It writes `VERIFIED`. Then the checkpoints at $2^{21}, 2^{23}, 2^{25}, 2^{27}$ are deleted (their SHA-256s stay in DONE and the deletions are listed in VERIFIED). The five even-power checkpoints are kept.
- **Merged, on CPU:**
  - all 160 concepts regenerate, with certified ranks and recomputed $K_{\text{bits}}$;
  - held-out documents regenerate;
  - oracle and mixture code lengths agree with the independent log-space forward algorithm to $10^{-9}$;
  - all 918 shards carry a VERIFIED marker matching their DONE file;
  - the analysis reruns to identical point estimates and gate verdicts;
  - the independent Tobit fitter agrees.

## 7. Compute, hardware and the cost cuts

- **Estimate:** about 4–5 minutes per S run and 9–10 per M run on an H200 or A100, about 105 GPU-hours in all. Prep (CPU) takes about 5 minutes.
- **Where:** SNAP skampere2 only. Other people's work and Laolu's other runs take precedence. skampere1 is not used at all, and on skampere2 a worker starts a job only on a GPU with ≥ 24 GB free and ≤ 30% utilization, re-checked before every job, at most 4 GPUs at once. If no GPU qualifies, the queue waits.
- **Cost cuts adopted** (only those with negligible effect on results):
  1. 1,024 evaluation documents per target instead of 2,048. The standard error of the all-position excess is about 0.001 bits against ε* = 0.1.
  2. Keeping 5 of 9 checkpoints after their replay. Storage only: every checkpoint is verified before deletion.
- **Rejected:** early stopping, fewer targets, fewer similarity levels, reusing 13-GPU's learning rates, and packing two runs per GPU. Each would change what is measured or add risk.
- **Storage:** outputs in `/dfs/scratch1/laoluog1/tdc_exp24/` (about 140 GB of kept checkpoints).

## 8. Disclosures and deviations

(Recorded here, with dates, by the pilot disclosure commit and any later deviation.)

## 9. Limits

- Synthetic languages only, two model sizes, three seeds, nine targets.
- Similarity is one constructed dial. Vocabulary overlap is not controlled inside the `copy` and `family` levels.
- The Task2Vec measure is reported, not validated as the right similarity measure here; Experiment 15 found measured alignment scores tracked hand-set α poorly in toy data.
- A 100-fold α range is still small next to real pretraining mixtures.

**TLDR-end:** [tdc: Expt 24 protocol] Cross a similarity dial for the non-target data (target copies with 20% or 50% of states re-drawn, same-family, disjoint vocabulary) with complexity (8–64 hidden states) and target share (0.01–1) on two transformer sizes, 918 runs trained to 2^28 tokens, and test with censoring-aware data bills whether similarity decides whether alignment or complexity dominates (G1–G5).
