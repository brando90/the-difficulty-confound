# Protocol: Can measured alignment recover the true alignment? (`15_alignment_alpha_recovery`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/15_alignment_alpha_recovery/PROTOCOL.md>

Status: DRAFT design, 10-18-2026, from Brando Miranda's complexity-versus-alignment idea. The running agent may revise it, then must freeze it in its own commit before any gated run; after the freeze nothing in Sections 2–5 changes.

Program overview: [complexity versus alignment, Experiments 11–16](../ideas/complexity_vs_alignment/README.md). Background on Task2Vec and complexity: [TASK2VEC_KOLMOGOROV.md](../ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md). Companion: [Experiment 14](../14_task2vec_vs_exact_complexity/PROTOCOL.md), whose frozen probe networks and Task2Vec code this experiment imports.

## 1. Scientific motivation

The [alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws) treats data–target alignment as a measured scaling coordinate. For a candidate pool $P$, a held-out target $D_{\text{target}}$ and a per-example score $s(x) \in [0,1]$, a training subset $S$ has token-weighted alignment

```math
\alpha(S) = \frac{\sum_{x\in S} t(x)\, s(x)}{\sum_{x \in S} t(x)} \qquad \text{(Eq. 2)},
```

where $t(x)$ is the number of training tokens in $x$. The main score is $s(x) = \tfrac12 \tilde s_{\text{SBERT}}(x) + \tfrac12 \tilde s_{\text{ZIP}}(x)$ (Eq. 3). Each component is min–max normalized over the pool. $\tilde s_{\text{SBERT}}$ is the maximum cosine similarity to target references under the Sentence-BERT model all-mpnet-base-v2. $\tilde s_{\text{ZIP}}$ is compression-ratio similarity under a 128 KB zstd dictionary trained on the references, adapted from ZIP-FIT ([Obbad, Mlauzi, Miranda, Schaeffer, Obbad, Bedi, Koyejo, arXiv 2410.18194](https://arxiv.org/abs/2410.18194)). Note that ZIP-FIT itself scores with gzip normalized compression distance (NCD), averaged over target examples (its Algorithm 1). The paper also tests a frozen-gradient selector in the spirit of LESS ([Xia, Malladi, Gururangan, Arora, Chen, ICML 2024, arXiv 2402.04333](https://arxiv.org/abs/2402.04333)). It then fits $L = a + bN^{-\beta} + cT^{-\gamma} - d\alpha$ (Eq. 1); here $N$ is fixed, so this is the program's `M_align_add`.

On real data nobody knows the true $\alpha$, so nobody can check whether the measured $\hat\alpha$ is right. In Experiments 11 and 12 the true $\alpha$ is set by hand in every training mix. This experiment asks four questions:

1. **Recovery.** Does $\hat\alpha$ rank mixes by true $\alpha$, comparably across targets?
2. **Complexity-neutrality (the difficulty-confound twist).** At fixed true $\alpha$, does $\hat\alpha$ drift with the target's complexity $K$? If simpler targets look more aligned, the paper's $\alpha$ partly measures complexity.
3. **Downstream use.** Can a calibrated $\hat\alpha$ replace the true $\alpha$ in Experiment 11/12's best law without losing held-out accuracy?
4. **Graded relevance.** When distractors are partly relevant, is "share of target data" even the right truth, or does a relevance-weighted $\alpha$ predict loss better?

**A fact that frames every gate.** Let the target hold share $\alpha$ of the mix, distractor $j$ hold share $\pi_j$, and let task $u$ have token-weighted mean score $m_u$. With equal mean example lengths across tasks, Eq. 2 gives

```math
\hat\alpha(S) = \alpha\, m_T + \sum_j \pi_j m_j = m_D + (m_T - m_D)\,\alpha, \qquad m_D = \tfrac{1}{1-\alpha}\textstyle\sum_j \pi_j m_j .
```

So, for a fixed target and distractor set, any score with $m_T \gt m_D$ ranks that target's mixes perfectly. The real questions are whether the intercept $m_D$ and slope $m_T - m_D$ are the same across targets, and in particular across $K$. Even a perfect per-example relevance oracle $s^\star(x) = \mathbf 1[y = f_T(x)]$ gives $m_T = 1$ and $m_j = \text{agree}_j$, the distractor's label agreement with the target. It therefore reports $\hat\alpha^\star = \alpha + \sum_j \pi_j\,\text{agree}_j \gt \alpha$. The graded arm (question 4) is built on this point.

**Competing predictions.**

- **Recovery holds and is complexity-neutral.** Then Eq. 2 is a trustworthy coordinate, and the alignment-scaling paper's $d$ is a clean alignment effect.
- **Recovery holds but drifts with $K$.** Simpler targets' references compress better and their label rules are more local, so compression and n-gram scores should separate simple targets' examples from distractors more easily. That raises $m_T - m_D$ for low $K$. The probe-embedding score may show the same bias if the probe represents simple label rules better. If $\hat\alpha$ rises as $K$ falls at fixed true $\alpha$, then in $L = a + cT^{-\gamma} - d\hat\alpha$ the fitted $d$ partly absorbs complexity. "Alignment dominates" would then be overstated exactly in the direction of the difficulty confound.
- **Recovery fails.** In these toys every task in a family shares one input distribution, and only the label separates a target example from a distractor example. Scores that look mainly at inputs (compression, n-grams, embeddings) may see almost nothing. Label-aware scores (the gradient score, Task2Vec distance) should do better. Real text is the opposite case: target and pool usually differ strongly in their inputs (topic, format). Failure here would mean the paper's $\alpha$ measures input-distribution alignment, not alignment of the input-to-label rule.

**Why these toys.** They are the only place where the true $\alpha$ is known exactly, the complexity of every target is exact, and Experiment 11/12 checkpoints give the downstream outcome for free.

**Disclosure.** Experiment 11/12 outcomes (losses, best law) are public before this protocol is frozen. The measured $\hat\alpha$ values and the graded-arm runs do not exist yet; the freeze must precede both.

## 2. Experimental design

### 2.1 Inputs (consumed, never modified)

- Experiment 11 (DFA = deterministic finite automaton; $K$ = minimal states $\in \{4,8,16,32\}$; 16 targets; $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$; 64 mixes) and Experiment 12 (DSL = domain-specific language; $K$ = exact shortest program $\in \{2,\dots,6\}$; 20 targets; 80 mixes). Files: `tasks.json`, `data/`, `mixes.json`, `metrics.csv`, `checkpoints/`, `manifest.json`, the frozen `PROTOCOL.md`, and the RESULTS file naming each experiment's best law on held-out cells.
- Experiment 14: frozen probes `probes/<family>_{14101,14202,14303,14404}.npz` (pretrained with a null task token) and `task2vec.py`, both imported read-only with SHA-256 checks against Experiment 14's manifest.

Mix shares are identical across training seeds, so each (target, $\alpha$) is one mix to score. The exact training multiset of each seed is reconstructed with Experiment 11/12's own assembly function, imported, and its per-task counts are asserted equal to `mixes.json`.

### 2.2 References, pools and serialization

- **Target references $R_T$:** 2,048 examples per target. They are the first 2,048 examples of the target's training file that no mix of any seed uses (from the reconstructed index sets). If fewer exist, draw fresh examples with the target's own generator and input sampler (seed 20261015) and log it.
  - Small input spaces make some content overlap between references and pool inevitable. An example whose input and label both match a reference is genuinely identical data. The exact-content overlap rate between $R_T$ and the pool is reported per target.
- **Strip the task token.** Serialize every example as a byte string with the task token removed: its input symbols, a separator, then its label (DFA: accept or reject; DSL: the 10 output bits). Each token maps to one printable byte (fixed table, saved). With the task token present, recovery is trivial; the leak control (Section 2.4) shows by how much. $t(x)$ = the number of tokens in the stripped sequence.
- **Scored units.** For each task, a fixed set $U_u$ of 4,096 training examples (fixed permutation, seed 20261015; the whole file if smaller), disjoint from $R_T$.
- **Pool for target $T$:** $P_T = U_T \cup \bigcup_{j \in D(T)} U_j$, where $D(T)$ is the distractor set that `mixes.json` records for $T$. Min–max normalization is computed over $P_T$, per target, per measure, per probe.
- **Stratified Eq. 2.** Every Experiment 11/12 mix is a stratified draw (fixed per-task counts $n_u(S)$), so

```math
\hat\alpha(S) = \frac{\sum_u n_u(S)\,\overline{t\,\tilde s}_u}{\sum_u n_u(S)\,\overline{t}_u},
```

  with means over $U_u$. This is Eq. 2 with each task's sum replaced by its unbiased sample mean. A test checks it against the exact Eq. 2 on full mixes for measure (c) on one seed, with tolerance $|\Delta\hat\alpha| \le 0.01$.

### 2.3 Measures

| ID | Measure | Definition | Role |
|---|---|---|---|
| a | ZIP, the paper's form | zstd dictionary trained on the 2,048 serialized references. Size = $\min(131072,\ 256\lfloor B_R/2560\rfloor)$ bytes, where $B_R$ is the reference byte count: one tenth of the reference bytes, about 2–4 KB here, versus 128 KB in the paper. Compression level 3; content-size, checksum and dictionary-ID frame fields off. $s_a(x) = \lvert x\rvert / C_{\text{dict}}(x)$ (compression ratio under the target dictionary). | half of (c) |
| a′ | ZIP, dictionary gain | $C_{\varnothing}(x)/C_{\text{dict}}(x)$, which removes how compressible $x$ is on its own | secondary |
| a2 | ZIP-FIT original | $1 - \frac{1}{128}\sum_j \mathrm{NCD}_{\text{gzip}}(x, r_j)$ over 128 random references; 512 scored examples per task | secondary |
| b1 | Embedding (stand-in for Sentence-BERT) | maximum cosine similarity to references of the mean-pooled final hidden states (after the final layer norm) of each Experiment 14 probe, null token | half of (c) |
| b2 | Model-free embedding | character 3-gram TF-IDF (term frequency–inverse document frequency) vectors fit on $R_T \cup P_T$, maximum cosine to references | secondary |
| **c** | **Paper's combined score** | $\tfrac12\tilde s_{b1} + \tfrac12\tilde s_{a}$ | **gated** |
| c2 | Model-free combined | $\tfrac12\tilde s_{b2} + \tfrac12\tilde s_{a}$ | secondary |
| d | Frozen gradient (LESS-style) | $\cos\big(g(x), \bar g_{R}\big)$. $g$ is the per-example gradient of the label cross-entropy (true labels, null token) with respect to all transformer-block and head parameters of an Experiment 14 probe, and $\bar g_R$ is the mean reference gradient. Embedding tables are excluded, so the score is not a token-overlap lookup. Unlike LESS: one checkpoint, no warm-up training, no Adam preconditioning, no low-rank adapters, no random projection. | secondary |
| e | Task2Vec mix-to-target distance (set-level, not Eq. 2) | cross-diversity $c(S)$ = mean symmetric Task2Vec distance (Experiment 14's E3 pipeline) over $8\times 8$ pairs of 256-example batches: 8 interleaved batches from $S$ and the 8 disjoint 256-example chunks of $R_T$. $\hat\alpha_e = \mathrm{clip}\big((c_0 - c(S))/(c_0 - c_1), 0, 1\big)$, where $c_0$ uses the $\alpha=0$ mix (distractors only, same relative shares) and $c_1$ uses 8 batches from $U_T$. | secondary |
| e2 | Task2Vec, fixed head | as (e), but one head is trained on $R_T$ and reused for every batch. With weights fixed, the Fisher information is linear in the mix ([research note](../ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md), Section 4(c)). | secondary |
| oracle | Label-agreement oracle | $s^\star(x) = \mathbf 1[y = f_T(x)]$ from the target's definition | reference, not a candidate |

For the probe-based measures (b1, c, d, e, e2), $\hat\alpha$ is computed per probe; the point estimate is the mean over the 4 probes. The paper's exact "compression-ratio similarity" formula is not public. If the running agent can read the paper's code, it uses that formula for (a) and records the source. Otherwise it uses the literal reading above, and (a′) covers the other reading.

### 2.4 Controls

- **Task-token leak (positive control):** (a), (b1) and (c) recomputed with the task token kept. Expect near-perfect recovery. The size of the gap to the stripped version is the reason the stripping rule exists.
- **Permuted-label references:** (a), (b1), (c) and (d) with the references' labels randomly permuted. A measure that still recovers $\alpha$ is reading something other than the input-to-label rule.
- **Input-only:** (a), (b1) and (c) on input tokens alone. With a shared input sampler this should carry no signal. The total-variation distance between target and distractor input distributions is reported, so this expectation is checkable.
- **Random score:** $s \sim U(0,1)$ per example (seed 20261015), giving the flat baseline.

### 2.5 Graded-relevance arm (new training, CPU)

- **Targets:** 4 DFA targets from Experiment 11, 2 at $K = 8$ and 2 at $K = 16$: the first in `tasks.json` order with acceptance rate in $[0.40, 0.60]$. If too few qualify, draw fresh ones with the Experiment 11 generator (seed 20261015) and log it. Balanced targets make chance agreement about 0.5, so the relevance correction below is nearly linear in agreement.
- **Distractors by rewiring:** for each target and each $r \in \{1, 2, 4, \text{all}\}$, 2 distractors (8 per target, 32 in total).
  - Pick $r$ (state, symbol) pairs uniformly without replacement from the target's minimal-DFA transition table, and send each to a uniformly random different state. "All" redraws every transition and keeps the state count and accept set.
  - Re-minimize with Hopcroft's algorithm and record the new $K$.
  - Reject and redraw (logged; at most 1,000 tries) if the language equals the target's or another distractor's, or if the acceptance rate is outside $[0.30, 0.70]$.
- **Relevance.** $\text{agree}_j = \Pr_{x \sim \text{target inputs}}[f_j(x) = f_T(x)]$, computed exactly by dynamic programming over the product automaton if Experiment 11's sampler is uniform given length; otherwise by $2^{20}$ Monte Carlo draws (standard error $\le 0.0005$), with the method recorded. Relevance is chance-corrected (Cohen's kappa):

```math
\rho_j = \max\!\Big(0,\ \frac{\text{agree}_j - a^0_j}{1 - a^0_j}\Big),\quad a^0_j = p_T p_j + (1-p_T)(1-p_j),\qquad \alpha_{\text{soft}} = \alpha_{\text{share}} + \sum_j \pi_j \rho_j .
```

  When $p_T = \tfrac12$, $\rho_j = \max(0, 2\,\text{agree}_j - 1)$, the uncorrected form proposed in the draft. We also record $\alpha_{\text{agree}} = \alpha_{\text{share}} + \sum_j \pi_j\,\text{agree}_j$, the oracle's value.
- **Construction check** (in `--check-only`, before any training): mean $\rho$ must fall strictly as $r$ rises, mean $\rho \ge 0.60$ at $r = 1$, and mean $\rho \le 0.20$ at $r = $ all. Otherwise redraw all distractors with the next seed (logged).
- **Mixes (5 per target), designed so $\alpha_{\text{share}}$ and $\alpha_{\text{soft}}$ are far from collinear.** The $\alpha_{\text{soft}}$ column is illustrative, assuming mean $\rho$ = 0.85, 0.7, 0.45, 0.05 for $r$ = 1, 2, 4, all.

| Mix | $\alpha_{\text{share}}$ | Distractors (equal shares) | Illustrative $\alpha_{\text{soft}}$ |
|---|---|---|---|
| m1 | 0.15 | 2 × ($r$ = all) | 0.19 |
| m2 | 0.15 | 2 × ($r$ = 1) | 0.87 |
| m3 | 0.15 | all 8 | 0.59 |
| m4 | 0.40 | 2 × ($r$ = all) | 0.43 |
| m5 | 0.40 | 2 × ($r$ = 2), 2 × ($r$ = 4) | 0.745 |

- **Training:** Experiment 11's training function, imported unchanged: same architecture (2 blocks, width 64), optimizer, schedule, $T$ checkpoints, fresh-versus-repeated data regime, and a task token on every training example. The task-token table is sized for the 36 graded tasks. Excess loss $\varepsilon$ is measured on the target's Experiment 11 test split at every checkpoint.
- **Seeds 15101, 15202, …, 15808:** 4 targets × 5 mixes × 8 seeds = **160 runs**. The draft proposed 4 seeds (80 runs); 8 seeds follow the program default and give the seed-block bootstrap a usable lower bound. Pre-declared fallback, decided only from the pilot's timing and before the freeze: if 160 runs project to more than 90 minutes on 4 workers, use seeds 15101–15404 (80 runs) and say so.
- **Pilot:** seed 15909 (never reused).
- **Graded-arm $\hat\alpha$:** measures (a)–(e) are computed for the 20 graded mixes exactly as above. References are 2,048 fresh target examples (seed 20261015); the pool is the target plus its 8 distractors.

### 2.6 Downstream law (for Gate 3)

For each family, use the law that Experiment 11 (or 12) reports as best on its held-out cells, with Experiment 11/12's fit and held-out cells, fitter (imported `laws.py`), scale for root-mean-square error (RMSE), and 8 training seeds. Fit and score three versions:

- true $\alpha$;
- calibrated $\hat\alpha_c$ (gated);
- raw $\hat\alpha_c$ (reported).

**Calibration:** an increasing isotonic regression of true $\alpha$ on $\hat\alpha_c$ over (target, $\alpha$) pairs whose $K$ level and $\alpha$ level are both outside the held-out levels, with linear interpolation between steps and clipping beyond the ends. Held-out rows never touch the calibrator (a test asserts this). Calibration uses the true $\alpha$ of fit cells, which a real-data user would lack. That is why raw $\hat\alpha$ is reported next to it.

### 2.7 Summary of the design

| Component | DFA (Experiment 11) | DSL (Experiment 12) |
|---|---|---|
| Targets / $K$ levels / mixes scored | 16 / 4 / 64 | 20 / 5 / 80 |
| References per target | 2,048 | 2,048 |
| Scored units per task | 4,096 | 4,096 |
| Probes (from Experiment 14) | 4 | 4 |
| Graded arm | 4 targets × 5 mixes × 8 seeds = 160 runs | none |

## 3. Measurements

1. Per-example scores for every measure, control and probe on every scored unit and reference (float32).
2. $\hat\alpha$ for every (family, target, $\alpha$, measure, probe), the oracle $\hat\alpha^\star$, and the token-weighted true share $\alpha^{\text{tok}}$ next to the example share $\alpha$.
3. Per target and measure: $m_T$, $m_D$, the separation $m_T - m_D$, the number of distinct score values in the pool (resolution), and the reference-coverage rate (share of pool inputs that appear among the references).
4. Graded arm: distractor definitions, $K$, agreement and $\rho$, $\alpha_{\text{share}}$, $\alpha_{\text{soft}}$ and $\alpha_{\text{agree}}$ per mix, training metrics, and checkpoints in the shared-contract format.
5. Gate 3 fits and forecasts under true, calibrated and raw $\hat\alpha$.

## 4. Analysis plan

**Bootstrap.** 20,000 draws, analysis seed 29101526, 95% percentile intervals. "LB" and "UB" are the 2.5th and 97.5th percentiles.

- Gates 1–2: resample targets with replacement within each $K$ level, and probe seeds with replacement.
- Gate 3: resample the 8 training-seed blocks (probe-averaged $\hat\alpha$).
- Gate 4a: resample the graded seed blocks.
- Gate 4b: resample targets and probes.

**Gate 1:** within each $K$ level, Spearman $\rho$ between $\hat\alpha$ and true $\alpha$ over the 16 (target, $\alpha$) points (4 targets × 4 $\alpha$), with average ranks for ties.

**Gate 2:** per family, regress $\hat\alpha_{T,\alpha} = c_\alpha + s \log_2 K_T + e$, with one intercept per $\alpha$ level, over all scored mixes. $s$ is the change in $\hat\alpha$ per doubling of $K$ at fixed true $\alpha$. Reported next to it: the same slope for the oracle and for $\hat\alpha - \hat\alpha^\star$, which separates measure bias from $K$-dependence built into the distractor construction.

**Gate 3:** relative change $\Delta = \mathrm{RMSE}(\hat\alpha_{\text{cal}})/\mathrm{RMSE}(\alpha) - 1$ on Experiment 11/12 held-out cells.

**Gate 4a:** leave-one-mix-out over m1–m5. In each fold, fit the best Experiment 11 law (fitted the way Experiment 11 fits it) on the 4 remaining mixes of all 4 targets, excluding the largest $T$. Score the left-out mix at every $T$, and the largest $T$ of every mix. Compare the pooled held-out RMSE with $\alpha = \alpha_{\text{soft}}$ against $\alpha = \alpha_{\text{share}}$. $\alpha_{\text{agree}}$ is reported as a third predictor.

**Reported for every measure:** Gate 1 and Gate 2 statistics, the affine decomposition $(m_D, m_T - m_D)$ per target versus $K$, the leak, permuted-label, input-only and random controls, and the independent refit.

**Fitting** uses the imported Experiment 11/12 fitter; `independent_fit.py` must agree to $10^{-4}$ relative.

## 5. Prospective pass/fail gates

Each gate ends `SUPPORTED` or `NOT_ESTABLISHED`, is reported with equal prominence, and is decided on measure (c) unless stated. Every other measure's numbers appear in the same table, without a verdict. Primary: Gates 1–3. Secondary: Gates 4a–4b.

1. **Gate 1 (primary): recovery.** `SUPPORTED` iff, in every $K$ level of both families (4 DFA levels and 5 DSL levels), Spearman $\rho(\hat\alpha_c, \alpha) \ge 0.90$ with 95% LB $\ge 0.80$.
2. **Gate 2 (primary): complexity-neutrality.** `SUPPORTED` iff, in each family, the 95% interval of the slope $s$ lies entirely inside $[-0.02, +0.02]$ per doubling of $K$. Prediction: at risk. The compression half may fail because simpler targets' references compress better and their label rules are more local. A `NOT_ESTABLISHED` result with $s \lt 0$ while the oracle slope is inside the band means the paper's $\alpha$ partly measures (low) complexity. The report must then state the consequence for the alignment-scaling law: $d$ in $L = a + cT^{-\gamma} - d\alpha$ absorbs part of a complexity effect.
3. **Gate 3 (primary): downstream non-inferiority.** `SUPPORTED` iff, in each family, the relative held-out RMSE change $\Delta$ from calibrated $\hat\alpha_c$ has point estimate $\le +0.10$ and 95% UB $\le +0.10$, with 0 invalid forecasts (nonfinite or below the floor).
4. **Gate 4a (secondary): graded relevance predicts loss.** `SUPPORTED` iff the law with $\alpha_{\text{soft}}$ reduces pooled held-out RMSE relative to $\alpha_{\text{share}}$ by $\ge 10\%$ with 95% LB $\gt 0\%$, with 0 invalid forecasts.
5. **Gate 4b (secondary): measured alignment tracks graded relevance.** On the 20 graded mixes: `SUPPORTED` iff $\rho(\hat\alpha_c, \alpha_{\text{soft}}) - \rho(\hat\alpha_c, \alpha_{\text{share}}) \ge 0.10$ with 95% LB $\gt 0$. Both Spearman values are reported for every measure.

## 6. What each outcome would mean

| Outcome | Interpretation |
|---|---|
| G1, G2, G3 `SUPPORTED` | On these toys the paper's measured $\alpha$ recovers the true $\alpha$, is not secretly a complexity measure, and can replace it in the law. |
| G1 `SUPPORTED`, G2 `NOT_ESTABLISHED`, oracle slope inside band | Measured alignment is biased by target complexity. Laws fit on measured $\alpha$ mix the two effects, and C2 ("alignment dominates") cannot be read from such fits without a complexity covariate. |
| G2 `NOT_ESTABLISHED` and oracle slope also outside band | The toy's distractor agreement varies with $K$. Read the measure's bias from the slope of $\hat\alpha - \hat\alpha^\star$. |
| G1 `NOT_ESTABLISHED` for (c), `SUPPORTED`-level numbers for (d) or (e) | Similarity and compression scores miss conditional (input-to-label) alignment; optimizer-aware or Fisher scores catch it. On real data, input-distribution alignment and rule alignment should be measured separately. |
| Leak control near-perfect, stripped scores near the random baseline | Recovery came only from identity cues. On real data the analog is template or format cues, which can make $\hat\alpha$ look excellent while measuring nothing about the task. |
| Permuted-label references recover as well as true ones | The measure ignores labels and reads the input distribution only. |
| G3 `SUPPORTED` but raw $\hat\alpha$ much worse | Measured $\alpha$ is usable only after calibration, which real data cannot provide. Report the raw number as the realistic one. |
| G4a `SUPPORTED` | Partially relevant data acts as partial alignment, so "share of target data" understates alignment. |
| G4a `NOT_ESTABLISHED`, $\alpha_{\text{share}}$ law fits | Only exact-target data counts here; near-miss tasks help no more than unrelated ones. |
| Both graded laws fail, m2 worse than m1 | Near-miss distractors interfere (conflicting labels on shared structure). Relevance is not a monotone benefit. |
| G4b `SUPPORTED` | The measured $\alpha$ is closer to graded relevance than to share, which is what a per-example relevance score should measure. |

## 7. Compute estimate and hardware

CPU only: 4 workers × 1 torch thread, RAM well under 12 GB. Scoring estimates assume 1,000–3,000 per-example gradients per second per core and about 0.3 s per 256-example Task2Vec batch embedding. The pilot measures both.

| Stage | Work | Wall time on 4 workers |
|---|---|---|
| Reconstruct mixes, references, scored units | 36 targets | about 5 min |
| (a), (a′), (b2), (c), (c2), oracle | about 370k examples | about 5 min |
| (a2) gzip NCD | about 45M pair compressions | about 5 min |
| (b1), (d) per probe | about 370k examples × 4 probes | about 10–20 min |
| (e), (e2) | (96 + 120) sets + 36 reference sets, × 8 batches × 4 probes × 2 variants ≈ 16k batch embeddings | about 20 min |
| Controls | (a), (b1), (c), (d) reruns | about 10 min |
| Graded arm | 160 runs at Experiment 11's per-run time (about 60–90 s on 1 thread) | about 40–60 min |
| Laws, bootstrap, verify | | about 15 min |
| **Total** | | **about 2 h** (range 1.5–3 h) |

Hard wall-clock cap in `config.json`: `max_wall_seconds = 14400` (4 h). Artifacts: about 0.1 GB of scores plus about 0.5 GB of graded-arm checkpoints, under 2 GB.

## 8. Artifacts and verification

Under `experiments/15_alignment_alpha_recovery/expt_v1/results/`:

- `references.npz` and `scored_units.npz` (indices and content hashes);
- `scores/<family>_<measure>_<probe>.npz`;
- `alpha_hat.csv`, `decomposition.csv`, `controls.csv`;
- graded arm in the shared contract: `graded/tasks.json`, `data/`, `mixes.json`, `metrics.csv`, `checkpoints/<seed>_<target>_<mix>/T_<T>.npz`;
- `laws.json`, `gates.json`, `bootstrap.npz`, figure `alpha_recovery.png`;
- `manifest.json` and `ARTIFACT_SHA256.json`.

`verify.py`:

- re-reconstructs every mix and asserts the counts;
- re-checks that references and scored units are disjoint and that no task token appears in any stripped serialization;
- recomputes all (a), (a′) and (b2) scores exactly, and a seeded 10% sample of (b1), (d) and (e) scores (maximum relative difference $\le 10^{-6}$);
- recomputes every $\hat\alpha$ from saved scores to $10^{-12}$;
- recomputes graded agreement exactly and cross-checks it against $2^{16}$ Monte Carlo draws (within 0.01);
- replays every graded-arm checkpoint's held-out loss to `max_abs_diff < 1e-9` bits;
- recomputes $K$ of every graded distractor;
- refits all laws with `independent_fit.py` to $10^{-4}$ relative;
- prints `PASS` or the first failure.

## 9. Scope and limits

- Results hold for these toys, these probes, and Eq. 2 as implemented. They do not show that the paper's $\alpha$ is right or wrong on its real-text grid.
- $K$ is exact complexity in a stated language, not universal Kolmogorov complexity.
- The graded arm covers only the DFA family and two $K$ levels.
- Isotonic calibration uses the true $\alpha$ on fit cells, which real data never provides; raw $\hat\alpha$ is the realistic number.
- Per-example scores on 10–20-byte examples sit near the resolution limit of a compressor. The resolution diagnostic says how close.

### What transfers to the real-text pipeline and what does not

| Transfers | Does not transfer |
|---|---|
| The affine decomposition $\hat\alpha = m_D + (m_T - m_D)\alpha$, and the warning that per-example relevance scores report "share of target-consistent data", not target share. | Absolute recovery numbers: real examples are hundreds to thousands of tokens, so compression and embedding scores have far more resolution. |
| The complexity-neutrality check: rerun the Gate 2 slope on real data by regressing $\hat\alpha$ on an available complexity proxy (for example, the Experiment 14 $\hat K$ if it passes there) at fixed construction quantile. | The 2–4 KB dictionary versus the paper's 128 KB dictionary trained on large text references. |
| Stripping identity cues: the real analog of the task token is the target benchmark's template or formatting, which should be normalized before scoring. | Sentence-BERT (all-mpnet-base-v2) semantics versus a 0.1 M-parameter probe's hidden states. |
| Calibration and raw-versus-calibrated reporting for any law that uses $\hat\alpha$. | The shared input distribution: real targets and pools differ mostly in inputs, where these scores are strongest. |
| Graded relevance: if Gate 4a passes, a chance-corrected relevance-weighted $\alpha$ is the better coordinate. | Exact agreement: on real data relevance has no ground truth, and only its proxies can be compared. |

**TLDR-end:** [tdc: Expt 15 design] Score every Experiment 11/12 training mix, with task tokens stripped, using the alignment-scaling paper's Eq. 2 under compression, probe-embedding, combined, frozen-gradient and Task2Vec scores, to test whether measured alignment recovers the hand-set $\alpha$ within each complexity level, stays flat across $K$, and can replace true $\alpha$ in the best law. A new 160-run graded-relevance arm tests whether chance-corrected relevance predicts loss better than target share.
