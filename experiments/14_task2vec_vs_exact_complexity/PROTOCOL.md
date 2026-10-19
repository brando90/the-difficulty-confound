# Protocol: Does Task2Vec track exact task complexity? (`14_task2vec_vs_exact_complexity`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/14_task2vec_vs_exact_complexity/PROTOCOL.md>

Status: FROZEN 10-19-2026, before any gated run (timing pilot disclosed in Section 2.8; the pre-declared fallback fired, so diversity coefficients use 8 batches). Drafted 10-18-2026 from Brando Miranda's complexity-versus-alignment idea. After the freeze nothing in Sections 2–5 changes; deviations are logged in CKPT_task2vec_complexity.md.

Program overview: [complexity versus alignment, Experiments 11–16](../ideas/complexity_vs_alignment/README.md). What Task2Vec does and does not prove about Kolmogorov complexity is in the research note [TASK2VEC_KOLMOGOROV.md](../ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md); this protocol does not repeat that argument, it tests its predictions.

## 1. Scientific motivation

Brando: "task2vec is basically a way for us to make the toy experiments eventually move to real data experiments." On real fine-tuning data (for example, math subsets) neither the target's complexity $K$ nor the training mix's alignment $\alpha$ is known. Task2Vec ([Achille et al., ICCV 2019, arXiv 1902.03545](https://arxiv.org/abs/1902.03545); reference code [awslabs/aws-cv-task2vec](https://github.com/awslabs/aws-cv-task2vec)) embeds a task as the filter-averaged diagonal of the Fisher information matrix (FIM) of a fixed probe network, and its paper says "the norm of the embedding correlates with the complexity of the task." Brando's diversity coefficient ([Miranda, Lee, Sundar, Casasola, Schaeffer, Obbad, Koyejo, "Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data", arXiv 2306.13840](https://arxiv.org/abs/2306.13840); maintained code [brando90/beyond-scale-div-coeff](https://github.com/brando90/beyond-scale-div-coeff)) is the expected cosine distance between Task2Vec embeddings of pairs of batches drawn from a dataset, with GPT-2 as the probe:

```math
\widehat{\mathrm{div}}(D) = \mathbb E_{B_1, B_2 \sim D}\; d\big(\vec f_{B_1}, \vec f_{B_2}\big).
```

If either quantity is to stand in for $K$ on real data, it must first track $K$ where $K$ is known exactly. Experiments 11 and 12 supply exactly that: tasks whose complexity in a fixed description language is computed exactly (minimal deterministic finite automaton (DFA) states by Hopcroft's algorithm; shortest program in a 12-primitive domain-specific language (DSL) by exhaustive search), with data, mixes and trained checkpoints saved under the shared output contract. This experiment trains no new task models. It pretrains small probe networks, embeds every Experiment 11/12 task, and asks four questions.

**Which program claims this serves.** It does not test C1–C3 directly. It tests whether a measurable proxy $\hat K$ could carry C1 and C3 (complexity raises the data bill; coefficient versus exponent) to data where $K$ is unknown.

**Competing predictions for the Task2Vec norm.**

1. **Norm tracks $K$.** The FIM is related to task complexity in the information-complexity framework of [Achille, Paolini, Mbeng, Soatto, "The Information Complexity of Learning Tasks, their Structure and their Distance", arXiv 1904.03292](https://arxiv.org/abs/1904.03292). If that relation survives the diagonal, filter averaging and a frozen probe, $\lVert F \rVert$ rises with $K$ in both families.
2. **Norm tracks probe-relative uncertainty, not $K$.** Task2Vec's own derivation (Sect. 2.2) shows that, for a fixed feature extractor, the norm scales with $\mathbb E[(y-p)^2] = p(1-p)$, the head's predictive uncertainty. It then follows label entropy, label noise and whatever the probe's features happen to represent. Transformers find high-sensitivity functions hard even when they are short ([Hahn 2020, arXiv 1906.06755](https://arxiv.org/abs/1906.06755); [Bhattamishra et al. 2023, arXiv 2211.12316](https://arxiv.org/abs/2211.12316)), so a parity-like low-$K$ task can have a large norm. Under this reading, Task2Vec measures difficulty for this probe, which is exactly the confounded quantity this repository studies.
3. **Norm predicts learning cost better than $K$.** Even if the norm is not $K$, it may predict how much data a task needs, $T^\star(\varepsilon^\star)$, because both reflect how well generic features fit the task.

**Prediction for the diversity coefficient.** Cosine distance discards the norm, so the diversity coefficient should measure coverage (how many distinct tasks a dataset mixes, and how different they are), not the complexity of any one task. This matches the diversity paper's finding that the coefficient "increases as the number of latent concepts increases" in the generative in-context learning (GINC) hidden Markov model (HMM) data. We therefore predict it tracks the mixture's task-distribution entropy and is nearly flat in the target's $K$.

**Why these toys.** Only here can $K$ be exact, varied on purpose, and crossed with label balance and sensitivity, all of which are recorded per task by Experiments 11 and 12. The input distribution is the same for every task in a family (the same sampler), so the task differs only in its labels.

**Disclosure.** Experiments 11 and 12 will have finished before this protocol is frozen, so their learning curves (and hence the $T^\star$ outcomes used in Gate 3) are already public. The predictors (all Task2Vec embeddings and diversity coefficients) do not exist yet. The freeze must happen before the first embedding is computed.

## 2. Experimental design

### 2.1 Inputs from Experiments 11 and 12 (consumed, never modified)

Read from each experiment's `expt_v1/results/` (in `experiments/11_dfa_exact_complexity_vs_alignment/` and `experiments/12_dsl_shortest_program_complexity_vs_alignment/`, or under `experiments/archive/` once archived): `tasks.json`, `data/<task_id>_{train,test}.npz`, `mixes.json`, `metrics.csv`, `manifest.json`. The model class, tokenizer or vocabulary, input sampler, task generator and exact-$K$ code are imported from Experiment 11/12 source files, and their SHA-256 hashes must equal those in the Experiment 11/12 `manifest.json`.

Experiment 11: $K$ = minimal DFA states $\in \{4, 8, 16, 32\}$, 4 targets and 4 distractors per level. Experiment 12: $K$ = exact shortest program length in the 12-primitive DSL $\in \{2, 3, 4, 5, 6\}$, 4 targets and 4 distractors per level. Both use $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$, a 2-layer transformer of width 64, and a task token on every example.

### 2.2 New tasks (task-sampling seed 20261014)

All new tasks are drawn with the Experiment 11/12 generator, imported, at the same $K$ levels, and must be pairwise distinct from every Experiment 11/12 task and from each other: no two DFAs may accept the same language (compare canonical minimal DFAs), and no two DSL tasks may compute the same function (compare full truth tables). Exact $K$ is recomputed for every new task with the imported code. If `tasks.json` already lists tasks with role `probe_pretrain` or `fresh`, use those (and check counts and disjointness) instead of drawing new ones.

| Role | DFA family (4 $K$ levels) | DSL family (5 $K$ levels) |
|---|---|---|
| Probe-pretraining tasks (16 per level) | 64 | 80 |
| Experiment 11/12 targets (4 per level) | 16 | 20 |
| Experiment 11/12 distractors (4 per level) | 16 | 20 |
| Fresh tasks for embedding (16 per level) | 64 | 80 |
| **Tasks embedded** (targets + distractors + fresh) | **96** | **120** |
| Label-noise arm: first 4 fresh tasks per level × noise rates $\{0.10, 0.25\}$ | 32 | 40 |

Every embedded task uses a fixed set of 2,048 examples: the first 2,048 of its training file under a fixed permutation (seed 20261014), or 2,048 freshly sampled examples for new tasks, from the same input sampler. The DSL family has only 512 training inputs per task (Experiment 12), so its 2,048 examples are 4 passes over those 512 inputs in a fixed order, as produced by Experiment 12's `data.example_stream`; this is the same for every DSL task, so the count stays fixed. Fixing the count matters because the Task2Vec norm depends on the number of samples (Task2Vec Sect. 3.1).

### 2.3 Probe networks

- **Architecture.** Exactly the Experiment 11 model for the DFA family and the Experiment 12 model for the DSL family (2 transformer blocks, width 64; about 0.1 M parameters with a 4× feed-forward layer; the exact count is read from the imported model and recorded). One permitted change: if the source model ties its input embedding to its output layer, the probe unties them, so that the output layer is a separate "head" module that Task2Vec can retrain while the feature extractor stays fixed.
- **Task token and the null token.** Every Experiment 11/12 example carries a task token. Task2Vec must not see it, or the embedding is a lookup of the task identity. We add one reserved null token. During pretraining, each example keeps its task token with probability 0.5 and gets the null token otherwise ("task dropout"), so the probe learns a meaningful null-token pathway. Every embedding in this experiment uses the null token. This mirrors real data, where GPT-2 is not told which task a batch comes from.
- **Pretraining.** The Experiment 11 (or 12) optimizer, learning-rate schedule and batch size; $2^{16} = 65{,}536$ fresh examples drawn uniformly over that family's probe-pretraining tasks (one pass, no repeats, for the DFA family; for the DSL family the 80 probe tasks have only 40,960 distinct training pairs, so this is about 1.6 passes over them, disclosed in the report); then frozen.
- **Seeds.** Four probes per family, seeds 14101, 14202, 14303, 14404, so every result averages over probes. Four, not eight: the bootstrap's main unit here is the task (96 or 120 per family), and each extra probe adds about 25 minutes of embedding. One extra randomly initialized, untrained probe per family (seed 14505) is a probe-swap control, as in the diversity paper's Appendix H.
- **Recorded:** each probe's held-out accuracy and loss per pretraining task with the task token and with the null token.

### 2.4 Task2Vec estimators

All estimators follow the reference implementation (`task2vec.py` in awslabs/aws-cv-task2vec, `method='montecarlo'`) as closely as a sequence model allows. The **head** is the probe's own output head, a linear map from the final hidden state to the label logits (Experiment 11's 2-way accept/reject head at the last position for the DFA family; Experiment 12's output head over the 10 output bits for the DSL family), initialized from the probe's trained head weights. Everything else is the **feature extractor** and is frozen. Gradients are taken through the full feature extractor.

| ID | Name | Head fit | Fisher labels | Gradient unit | Sampler | Role |
|---|---|---|---|---|---|---|
| E0 | Reference Monte Carlo | Adam, lr $4\times10^{-4}$, weight decay $10^{-4}$, 10 epochs of 10,000 draws, batch 64 (reference defaults) | sampled from the head's predictive distribution | squared gradient of the batch-mean loss, batch 64, 156 batches of one 10,000-draw epoch, averaged | class-balanced with replacement (reference `_get_loader`); uniform if a task has more than one label token | **primary** |
| E1 | Empirical FIM | as E0 | true labels | as E0 | as E0 | secondary |
| E2 | Per-example Monte Carlo | as E0 | sampled, 5 draws per example | per-example squared gradients over all 2,048 examples | none (each example once per draw) | secondary; the textbook diagonal estimator |
| E3 | Diversity-paper pipeline | full output layer fine-tuned with AdamW, lr $5\times10^{-5}$, eps $10^{-8}$, weight decay $10^{-4}$, 5 epochs, batch 8 (beyond-scale-div-coeff `_finetune_classifier`) | next token sampled at every position | squared gradient of the batch-mean loss, batch 8 | sequential | secondary for tasks; **primary for the diversity coefficient** |

Two notes. For Monte Carlo labels, the expected per-example gradient is zero, so E0's squared batch-mean gradient estimates the diagonal FIM divided by the batch size; E0 and E2 agree up to a constant in expectation, and the comparison checks that. With true labels (E1), the batch-mean gradient also contains the squared mean gradient, which is why E1 is not the primary.

Head fitting may run on cached final hidden states. That is exact because the feature extractor is frozen and the probe has no dropout; a unit test checks it.

**Aggregation (reference `extract_embedding`).** For every module except the head that has a `weight`, reshape the accumulated squared gradient to (rows, −1) and average over columns: one value per output unit of each linear layer, per row of each embedding table, and per element of each layer-norm weight. Concatenate the values in `named_modules()` order. Biases are ignored, as in the reference. The dimension is fixed by the architecture (about two thousand); it is computed and recorded at build time.

**Summaries.** Norm $= \lVert f \rVert_1$, the L1 norm, as in Task2Vec Fig. 2 (right). We also report $\lVert f \rVert_2$ and $\tfrac12 \sum_i \log(f_i + 10^{-12})$. The **symmetric Task2Vec distance** is the reference `cosine`: $d_{\text{sym}}(f_a, f_b) = d_{\cos}\big(f_a/(f_a+f_b+10^{-8}),\, f_b/(f_a+f_b+10^{-8})\big)$, elementwise. The beyond-scale-div-coeff code uses the same function. For averaging across probes, each probe's $\log \lVert f \rVert_1$ is z-scored within family; the task's score $\hat K_{\text{T2V}}$ is the mean of the four z-scores.

### 2.5 Diversity coefficient

- **Definition (diversity paper, Sect. 2.2, implemented as `get_diversity_coefficient`).** Draw 12 batches of 512 examples from the dataset, embed each batch with E3 under the same probe, and average $d_{\text{sym}}$ over the 66 unordered pairs (after the pre-declared fallback, Section 2.8: 8 batches and 28 pairs). The paper used 200 batches (and the code defaults to 600); 12 is what a 4-core CPU affords. The bootstrap interval over batch pairs is reported for every coefficient.
- **Interleaved batches (primary).** Each batch is an i.i.d. draw from the mix: pick a task by its share, then a uniformly random example of that task. This is the paper's interleaved-mixture protocol (Appendix I.6). Sequences from all tasks share a batch.
- **Single-task batches (secondary).** Pick a task by its share, then fill the whole batch from it. This is the paper's concatenated-dataset protocol (Appendix I.4). Here theory predicts $\widehat{\mathrm{div}} \approx \sum_{j \ne k} \pi_j \pi_k\, d_{\text{sym}}(F_j, F_k)$ plus a within-task noise floor. That sum is Rao's quadratic entropy $Q$ of the mix under task embeddings, computed from the E3 task embeddings.
- **Mixes scored.**
  - All Experiment 11/12 training mixes: 16 DFA targets × 4 $\alpha$ = 64, and 20 DSL targets × 4 $\alpha$ = 80, with shares from `mixes.json`. Shares are identical across training seeds, so one mix per (target, $\alpha$).
  - **Coverage mixes**, built only for this measurement (no training): $m \in \{1, 2, 4, 8, 16\}$ fresh tasks from a single $K$ level, uniform shares, 2 replicate draws per ($K$, $m$); DFA 4 × 5 × 2 = 40, DSL 5 × 5 × 2 = 50. At $m = 16$ both replicates use all 16 fresh tasks of that level and differ only in batch sampling. Coverage mixes cross entropy $H = \log_2 m$ with $K$ by construction. In the Experiment 11/12 mixes, $H$ is a function of $\alpha$ alone.
  - **Reference bounds (diversity paper, Sect. 2.4):** a lower-bound dataset in which every position is one fixed non-special token, and an upper-bound dataset of uniformly random tokens over the family vocabulary (task tokens excluded). Computed per family and probe.
- Interleaved: 234 mixes × 4 probes × 12 batches. Single-task: 90 coverage mixes × 4 probes × 12 batches. (8 batches each after the fallback, Section 2.8.)

### 2.6 Covariates and outcomes per task

From `tasks.json`: $K$, $K_{\text{bits}}$, acceptance rate or output-bit balance, sensitivity, visit-tail exponent (DFA). Derived:

- **Label entropy:** the binary entropy of the acceptance rate (DFA), or the mean per-output-bit binary entropy (DSL).
- **Head residual error:** the E0 head's training cross-entropy and error after fitting, the quantity Task2Vec Sect. 2.2 says the norm tracks.
- **Data requirement** $T^\star(\varepsilon^\star)$ for each target, from Experiment 11/12 `metrics.csv` at $\alpha = 1$. Per training seed, $T^\star$ is the first $T$ at which excess loss $\varepsilon \le \varepsilon^\star$, using log–log linear interpolation between adjacent checkpoints and the $\varepsilon^\star$ frozen in Experiment 11/12's protocol. A curve that never reaches $\varepsilon^\star$ is right-censored and gets $T^\star = +\infty$, which ranks above every finite value. The target's value is the median over seeds.
- **Prequential code-length proxy** for each target (report-only): the trapezoid integral of $\varepsilon$ over $T$ on the $\alpha = 1$ checkpoints, a learner-based computable comparison suggested by the research note.

### 2.7 Controls held fixed and how each confound is handled

| Confound | Control |
|---|---|
| Task token reveals identity | Null token for every embedding; probe pretrained with task dropout. |
| Sample count changes the norm | 2,048 examples per task, 512 per batch, everywhere. |
| Label balance and entropy | Class-balanced sampler in E0 (as the reference); label entropy is a covariate in Gate 2. |
| Sensitivity (parity-like hardness) | Sensitivity is a covariate in Gate 2. |
| Label noise at fixed $K$ | Label-noise arm, Gate 5. |
| Probe choice | 4 probe seeds, a random-init probe, and Kendall's W across probes. |
| Parameterization | Function-preserving rescaling check (Section 3). |
| Input distribution | One sampler per family, so the total-variation distance between task input distributions is 0 by construction (checked). |
| Train/test overlap | Only training examples are embedded; test files are never read except by Experiment 11/12 metrics. |

### 2.8 Pilot disclosure (10-19-2026, before the freeze)

**Timing pilot.** `run.py --pilot`, seed 14909, both families, 07:34–07:54 (19.5 min wall): one trained probe per family; 8 embedded tasks per family (evenly spaced in the K-sorted list) with all four estimators; 4 of them with E0 on the rescaled probe; two interleaved diversity coefficients per family (the first Experiment 11/12 training mix and the m = 16 coverage mix of the first level) at 12 batches. Per-unit times measured with 4 workers running at once: DFA 77.8 s per all-estimator task, 17.7 s per E0-only task, 4.27 s per E3 batch, 198 s probe pretraining; DSL 130.6 s, 35.0 s, 2.21 s, 143 s. Projection (core seconds / 4 + 0.25 h): **12.9 h at 12 batches, 11.8 h at 8 batches**.

**Decision.** The projection exceeds 2.5 h, so the pre-declared fallback applies: `diversity_batches` = 8 (28 pairs per coefficient) in `config.json`. Nothing else changes.

**Context for the timing.** The box was shared: other jobs, not visible to this run, kept the load average at 7–10 during the pilot and the CPU 95% busy after it ended. The pilot's processes received 683 CPU-seconds in 19.5 minutes, roughly a quarter of the 4 cores, so the per-unit times are several times the uncontended rate (development benchmarks on a free CPU: DFA 7 s, 3.6 s, 0.83 s; DSL 19 s, 6.7 s, 0.5 s, which project to about 2.1–2.4 h at 12 batches, close to the threshold). The rule is applied as written to the measured pilot. At the measured contended speed even 8 batches exceed the 4 h cap; the run proceeds in chunks of at most 60 minutes, and if the cap is reached first it stops and the report states what is missing, without extrapolation. The Section 7 per-unit estimates (about 1 s per task embedding) were optimistic: E2 needs 5 sampled labels for each of 2,048 examples, and E0/E1 differentiate the full feature extractor for 156 batches.

**Implementation choices made before the freeze (no results seen).**
- E0 and E1 use the same class-balanced draws and batches (E1 swaps in the true labels), so their difference isolates the label source.
- The rescaled probe reuses the original probe's draws and head-fit seed, so the rescaling check measures the parameterization alone.
- Diversity batches depend only on (task-sampling seed, family, mix, batch kind), so every probe sees the same batches.
- Gradients are exact but computed with `torch.func`: per-example gradients of the 2-way score with `vmap(grad)` for DFA (right-padded chunks of 128 read at each string's last position, exact because the model is causal), and one forward plus `vjp` per batch, vmapped over cotangent sets, for DSL. Batch gradients are combined from per-example rows with float32 sparse products; squares and sums are float64. A unit test checks E0–E3 against direct autograd for both families (relative difference below 1e-5).
- Only `weight` tensors are differentiated: biases are dropped by the aggregation (Section 2.4), so their gradients are never needed.
- The DSL family's 2,048 examples per task are 4 passes over its 512 training inputs (Section 2.2).
- Job order in the full run: all-estimator embeddings, then diversity, then E0-only embeddings (label noise first, then rescaled and random-init probes).

## 3. Measurements

1. Probe pretraining curves and held-out accuracy per pretraining task, with and without the task token.
2. For each embedded task × probe (4 trained + 1 random-init): E0 embedding. For the 4 trained probes, also E1, E2 and E3. Label-noise arm: E0 only. Each embedding is saved with its head's final training loss and error.
3. **Rescaling check** (parameterization sensitivity): for each trained probe, build a function-preserving copy that multiplies every query weight and bias by $c = 2$ and divides every key weight and bias by $c$, and does the same for value (×2) and attention-output (÷2) weights (exact in any transformer because attention logits are bilinear in query and key and the output is linear in value). Verify max output difference $\lt 10^{-5}$, then recompute E0 for all tasks.
4. Diversity coefficients (interleaved for all 234 mixes, single-task for the 90 coverage mixes), the lower and upper reference bounds, and Rao's $Q$.
5. Layer subsets: norm from block 1 units only and from block 2 units only (from saved embeddings, no recomputation).

## 4. Analysis plan

**Statistics.** Spearman $\rho$ with average ranks for ties. Partial Spearman: rank-transform every variable, regress the rank of $x$ and the rank of $y$ (each with an intercept) on the ranks of the covariates by ordinary least squares, and take the Pearson correlation of the two residual vectors. An independent second implementation (`independent_stats.py`) must agree to $10^{-10}$.

**Bootstrap.** 20,000 draws, analysis seed 29101426. 95% percentile intervals. "LB" is the 2.5th percentile.

- Gates 1, 2 and 5: resample tasks with replacement within each $K$ level (keeping every level's count), and probe seeds with replacement from the four. Recompute each task's $\hat K_{\text{T2V}}$ from the resampled probes.
- Gate 3: also resample Experiment 11/12 training-seed blocks with replacement and recompute each target's median $T^\star$.
- Gate 4: resample coverage-mix replicates within each ($K$, $m$) cell, batch pairs within mix, and probe seeds.

**Analyses.**

- G1–G3 per family for E0 (gated), then repeated for E1, E2, E3, the random-init probe, the rescaled probe, L2 and log-sum summaries, and layer subsets (all reported, none gated).
- **Probe-seed sensitivity:** Kendall's W across the 4 probes for the 96 or 120 task norms; mean pairwise Spearman between probes; the fraction of probe pairs that disagree on the order of the $K$-level mean norms.
- **Parameterization sensitivity:** Spearman between original and rescaled-probe norms; change in the Gate 1 $\rho$.
- **Mechanism:** Spearman of the norm with head residual error and with label entropy; Spearman of head residual error with $K$. If the norm follows head error, and head error follows $K$ only through sensitivity, prediction 2 holds.
- **Incremental value over exact $K$ (report-only, required):**
  - partial Spearman of norm with $T^\star$ given $K$ (and given $K$ plus label entropy and sensitivity);
  - leave-one-target-out root-mean-square error (RMSE) of $\log T^\star$ under $\log T^\star \sim \log_2 K$ versus $\log T^\star \sim \log_2 K + \log \lVert f \rVert_1$, with censored targets excluded and their count reported.
- **Diversity:** Spearman of $\widehat{\mathrm{div}}$ with $H$, $K$, $\alpha$ and Rao's $Q$ across coverage mixes (gated) and across Experiment 11/12 mixes (reported). Separately on the $\alpha = 1$ slice: whether single-task diversity rises with $K$, since within-task gradient variability could grow with complexity.
- **Mix-to-target distance:** handed to [Experiment 15](../15_alignment_alpha_recovery/PROTOCOL.md) (measure e), which owns the alignment gates. This experiment saves the probes and the Task2Vec code that 15 imports.

## 5. Prospective pass/fail gates

Each gate ends `SUPPORTED` or `NOT_ESTABLISHED`, is reported with equal prominence, and must hold in **both** families unless stated. Primary: Gates 1–3. Secondary: Gates 4–5.

1. **Gate 1 (primary): the Task2Vec norm tracks $K$.** On the 96 DFA and 120 DSL embedded tasks, with E0 and $\hat K_{\text{T2V}}$ from the L1 norm: `SUPPORTED` iff Spearman $\rho(\hat K_{\text{T2V}}, K) \ge 0.50$ with 95% LB $\gt 0.20$, in each family.
2. **Gate 2 (primary): not a label-balance or sensitivity artifact.** Same tasks: `SUPPORTED` iff the partial Spearman of $\hat K_{\text{T2V}}$ with $K$, controlling for label entropy and sensitivity, is $\ge 0.30$ with 95% LB $\gt 0$, in each family.
3. **Gate 3 (primary): the norm predicts learning cost.** On the 16 DFA and 20 DSL targets: `SUPPORTED` iff Spearman $\rho(\hat K_{\text{T2V}}, T^\star(\varepsilon^\star)) \ge 0.50$ with 95% LB $\gt 0$, in each family. The incremental value over exact $K$ is reported next to it and not gated.
4. **Gate 4 (secondary): the diversity coefficient measures coverage, not complexity.** Interleaved diversity on the coverage mixes (40 DFA, 50 DSL): `SUPPORTED` iff $\rho(\widehat{\mathrm{div}}, H) \ge 0.70$ with 95% LB $\gt 0.50$, **and** $\rho(\widehat{\mathrm{div}}, H) - \lvert\rho(\widehat{\mathrm{div}}, K)\rvert \ge 0.30$ with 95% LB $\gt 0$, in each family. Prediction: `SUPPORTED`. More distinct tasks make batch-level Fisher diagonals differ more from batch to batch. $K$ changes how hard each task is but not how many there are, and cosine distance discards the norm. The prediction can fail in one informative way: with interleaved 512-example batches, every batch holds every task in nearly the same proportions, so the coefficient may sit near its within-batch noise floor for every mix. The single-task variant is reported to show whether that happened.
5. **Gate 5 (secondary): label noise at fixed $K$ raises the norm.** On the label-noise arm: `SUPPORTED` iff the median over tasks of $\lVert f \rVert_1(25\%\ \text{noise}) / \lVert f \rVert_1(0\%)$ is $\ge 1.20$ with 95% LB $\gt 1.00$, in each family. Prediction: `SUPPORTED`, which would show the norm responds to something that leaves $K$ unchanged. A `SUPPORTED` Gate 5 is evidence against reading the norm as $K$, so the report must say so whatever Gates 1–3 show.

## 6. What each outcome would mean

| Outcome | Interpretation |
|---|---|
| G1, G2, G3 `SUPPORTED`; probe-seed W ≥ 0.8; rescaled-probe $\rho \ge 0.9$ | The norm is a usable, stable $\hat K$ on these toys and predicts learning cost. Proceed to the real-data handoff (Section 9). |
| G1 `SUPPORTED`, G2 `NOT_ESTABLISHED` | The norm–$K$ link runs through label entropy or sensitivity. Task2Vec then measures probe-relative difficulty, and using it as $\hat K$ would re-import the difficulty confound. |
| G1 and G2 `NOT_ESTABLISHED`, G3 `SUPPORTED` | The norm predicts data cost but is not complexity. It can enter a law as a learner-relative covariate, labelled as such, never as $K$. |
| G3's incremental value over $K$ positive | Task2Vec adds information about learning cost beyond exact description length (for example, sensitivity or how the probe's features fit the task). |
| G1 `SUPPORTED` in one family only | The relation depends on the description language or architecture; no general $\hat K$ claim. |
| Low W across probes, or the rescaling check flips rankings | The norm is too probe- or parameterization-dependent to carry across models; any real-data use must fix the probe and report this sensitivity. |
| G4 `SUPPORTED` | The diversity coefficient measures coverage; on real data it should be read as an "effective number of tasks", not a complexity or difficulty measure. |
| G4 `NOT_ESTABLISHED`, single-task variant tracks $Q$ | Batch construction, not the coefficient's meaning, washed out the signal; real-data use needs batch sizes and mixing stated. |
| G5 `SUPPORTED` | The norm rises with label noise at fixed $K$, as the research note predicts. It is at best an upper-bound-like proxy mixing $K$ with noise entropy. |
| Random-init probe matches trained probes | The signal is in the gradients of the input and label statistics, not in learned features. |

## 7. Compute estimate and hardware

CPU only: 4 worker processes × 1 torch thread, RAM well under 12 GB. Estimates assume about 1 s per E0, E1 or E2 task embedding, about 0.5 s per E3 batch embedding with cached head features, and 2–4 minutes per probe pretraining run. The disclosed pilot measures these before the freeze.

| Stage | Count | Single-core time | Wall time on 4 workers |
|---|---|---|---|
| Probe pretraining | 8 probes | about 25 min | about 8 min |
| Task embeddings (E0–E3, noise arm, rescaled, random-init) | DFA 4 × 512 + DSL 4 × 640 + 216 random-init ≈ 4,800 | about 80 min | about 20 min |
| Diversity (interleaved, single-task, bounds) | 11,232 + 4,320 + 192 ≈ 15,700 batch embeddings | about 130 min | about 35 min |
| Statistics and bootstrap | 20,000 draws per gate | about 10 min | about 10 min |
| **Total** | | | **about 75 min** (range 1–2 h) |

Hard wall-clock cap in `config.json`: `max_wall_seconds = 14400` (4 h). Pre-declared fallback, chosen only from the pilot's timing and before the freeze: if the projection exceeds 2.5 h, cut the diversity batches from 12 to 8 (28 pairs). Artifacts: about 0.3 GB (embeddings in float32, batch embeddings, probes).

## 8. Artifacts and verification

Under `experiments/14_task2vec_vs_exact_complexity/expt_v1/results/`:

- `tasks14.json` (new tasks with $K$, definitions and roles);
- `probes/<family>_<seed>.npz` (float32 weights, no pickle);
- `embeddings.npz` keyed by (family, task, probe, estimator, variant), with head losses;
- `batch_embeddings/<family>_<probe>.npz`;
- `diversity.csv`, `task_table.csv`, `gates.json`, `bootstrap.npz`, figure `task2vec_vs_K.png`;
- `manifest.json` (source and Experiment 11/12 SHA-256s, versions, wall time);
- `ARTIFACT_SHA256.json`.

`verify.py`:

- reloads every probe and recomputes its held-out pretraining loss to `max_abs_diff < 1e-9` bits;
- recomputes the exact $K$ of every new task with the imported Experiment 11/12 code;
- re-checks role disjointness;
- recomputes a seeded 10% sample of all embeddings, and every embedding behind a gate decision for one probe, with the same seeds and 1 thread, to a maximum relative coordinate difference $\le 10^{-6}$;
- recomputes every statistic and gate from saved embeddings with `independent_stats.py` to $10^{-10}$;
- prints `PASS` or the first failure.

## 9. Scope and limits

- $K$ is complexity in a stated language (minimal DFA states; shortest DSL program), not universal Kolmogorov complexity. A positive result says the Task2Vec norm tracks that quantity for these probes and tasks, nothing more.
- The probe is tiny and trained on the same task family. A GPT-2 probe on text differs in scale, tokenization and pretraining; the handoff below exists because passing here is necessary, not sufficient.
- E0 follows the reference `montecarlo` estimator; the Task2Vec paper's variational ("robust") Fisher and the β = 1 variational code length discussed in the research note are not computed here (cost). They are natural follow-ups if Gates 1–2 are borderline.
- $T^\star$ is known before the freeze (see Section 1); only the predictors are prospective.
- Eight batches per diversity coefficient (12 planned; the pre-declared fallback fired, Section 2.8) is far below the paper's 200. The intervals report the resulting noise; small differences between mixes should not be over-read.

### Handoff to real data

Before the Task2Vec norm is used as $\hat K$ on real fine-tuning data, all of the following must hold here:

- Gates 1, 2 and 3 `SUPPORTED` in both families.
- Gate 5 reported, and its effect size stated next to any $\hat K$ use.
- Kendall's W across probes $\ge 0.80$.
- Spearman between original and rescaled-probe norms $\ge 0.90$.
- The incremental value over exact $K$ in Gate 3 is non-negative.

If any of these fails, the norm may still be used, but only as a probe-relative difficulty covariate, labelled that way.

**Next experiment (sketch, not run).** Take math subsets with known structure: MATH ([Hendrycks et al. 2021, arXiv 2103.03874](https://arxiv.org/abs/2103.03874)) split into its 7 subjects × 5 difficulty levels (35 subsets), plus GSM8K ([Cobbe et al. 2021, arXiv 2110.14168](https://arxiv.org/abs/2110.14168)).

- **Embedding.** Use GPT-2 as the probe, exactly as in the diversity paper, and a fixed $n$ problems per subset ($n$ = the smallest subset's training size, stated).
- **Outcome.** For each subset, fine-tune one small model at a $T$ grid and measure $T^\star(\varepsilon^\star)$ on that subset's held-out problems. This is a GPU job on the Stanford InfoLab cluster (SNAP) or Sherlock.
- **Baselines.** Starting difficulty (base-model loss on the subset, the repository's $D_{m_0}$), answer length, and the MATH level label.
- **Gate.** Adding the Task2Vec norm cuts leave-one-subset-out RMSE of $\log T^\star$ by $\ge 10\%$ with LB $\gt 0$.
- **Leakage rule** ([predictability bridge](../ideas/predictability_bridge/README.md)): the norm is a starting-probe measurement and is allowed; $T^\star$ outcomes may not feed any predictor.

**TLDR-end:** [tdc: Expt 14 design] Embed every Experiment 11/12 task, plus 144 fresh tasks, with Task2Vec under 4 frozen probe networks to test whether the Fisher-embedding norm tracks exact complexity $K$ beyond label entropy and sensitivity and predicts the data a task needs, and whether Brando's diversity coefficient measures coverage rather than complexity; only if both pass may the norm serve as $\hat K$ on real math data.
