# Protocol: GINC-style hidden Markov model mixtures, complexity versus alignment with an exact floor (`13_ginc_hmm_complexity_vs_alignment`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/13_ginc_hmm_complexity_vs_alignment/PROTOCOL.md>

Status: DRAFT design, 10-18-2026, from Brando Miranda's complexity-versus-alignment idea. The running agent may revise it, then must freeze it in its own commit before any gated run; after the freeze nothing in Sections 2–5 changes.

Program overview: [complexity versus alignment, Experiments 11–16](../ideas/complexity_vs_alignment/README.md). Notation follows it: $\varepsilon$ excess loss in bits, $T$ training tokens seen, $\alpha$ target share, $K$ exact complexity in a stated description language, and the law family `M0_scale`, `M_align_eff`, `M_align_add`, `M_K_coef`, `M_K_exp`, `M_K_both` with parameters $A, \eta, \kappa, \gamma_0, \gamma_1$.

The experiment has two arms under one protocol: **13-small** (CPU, central processing unit; run by Brando's assistant, fixed model size) and **13-GPU** (Laolu or Brando on the SNAP (Stanford Network Analysis Project) graphics processing unit (GPU) nodes or Sherlock, three model sizes $N$).

## 1. Scientific motivation

**Hypothesis.** Brando's claim (10-18-2026) is that complexity is a real learning cost, but alignment dominates once a model has trained on aligned data. Experiments 11 and 12 test it on deterministic tasks, where the floor is 0. Experiment 13 tests it on stochastic sequence data of the kind in-context learning research uses, where the floor is not 0 but can be computed exactly. It tests C1 (complexity is real), C2 (alignment dominates) and C3 (coefficient or exponent), and adds one prediction only this setting can test: **alignment and complexity act at different context positions.**

**Why this toy.** GINC (Generative In-Context learning dataset; [Xie, Raghunathan, Liang and Ma 2021, "An Explanation of In-context Learning as Implicit Bayesian Inference", arXiv 2111.02080](https://arxiv.org/abs/2111.02080); code [p-lambda/incontext-learning](https://github.com/p-lambda/incontext-learning)) generates pretraining documents from a uniform mixture of hidden Markov models (HMMs), one "concept" per document, so a language model must infer the concept from context. Brando's diversity paper ([Miranda, Lee, Sundar, Casasola, Schaeffer, Obbad and Koyejo, "Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data", arXiv 2306.13840](https://arxiv.org/abs/2306.13840), Section 3.5 and Appendix K) used GINC to validate the diversity coefficient: on GINC datasets with 1 to 10,000 latent concepts at vocabulary sizes 50 and 150, the coefficient rose with the number of concepts and saturated, and with 5 and 5,000 concepts it rose with vocabulary size from 50 to 10,000. That work varied how many concepts a dataset mixes (coverage). This experiment varies the complexity of one concept and how much of the training data comes from it (alignment), on the same kind of data.

**What we take from GINC and what we change.** We keep the mixture-of-HMM-concepts structure, documents drawn from one concept each, a vocabulary shared across concepts, and no task token: the model is never told which concept a document comes from, so alignment cannot be read off a label. We drop GINC's factorial entity/property HMM, its memory-matrix emissions and its delimiter-separated prompts. In their place each concept is a sparse single-chain HMM, for two reasons: its description length has a closed form, and its minimal state count can be certified exactly (Section 2.2). Evaluation is next-token loss on held-out documents, not GINC's in-context accuracy on delimiter prompts.

**Two exact references.** For every held-out token we compute two predictive distributions exactly with the forward algorithm:

- the **oracle**, the true target concept's predictive $p^\star(x_t \mid x_{\lt t})$. Its code length is the floor $L^\star$;
- the **Bayes mixture under the training prior**, $p_{\text{mix},\alpha}(x_t \mid x_{\lt t}) = \sum_c w_c(x_{\lt t})\,p_c(x_t \mid x_{\lt t})$. Here the prior puts weight $\alpha$ on the target and $(1-\alpha)/D$ on each of the $D$ distractor concepts, and $w_c$ is the posterior after the prefix. This is the best next-token predictor for the training distribution. A model that learned its training data perfectly would converge to it.

That gives an exact decomposition of the excess loss:

```math
\varepsilon \;=\; \underbrace{L_{\text{model}} - L_{\text{mix},\alpha}}_{\varepsilon_{\text{learn}}:\ \text{not yet learned the training distribution}} \;+\; \underbrace{L_{\text{mix},\alpha} - L^\star}_{\varepsilon_{\text{id}}(\alpha):\ \text{identification cost under the training prior}} .
```

Over a whole document the identification cost is at most $\log_2(1/\alpha)$ bits, because $-\log_2 \sum_c \pi_c P_c(x_{1:t}) \le -\log_2 \alpha - \log_2 P^\star(x_{1:t})$. At $\alpha = 1$ it is exactly 0. So $\varepsilon_{\text{id}}$ is an exact, mechanistic version of an alignment penalty that survives infinite data. It is the closest thing this program has to the $-d\alpha$ term of the [alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws), and we report whether it is linear in $\alpha$ as that term assumes.

**Competing predictions.**

| Picture | Prediction in this experiment |
|---|---|
| Coefficient (minimum description length, MDL) | A concept with $n$ states has about $5n$ free probabilities, so the Bayes-mixture regret grows linearly in $n$: $\kappa \approx 1$ in $K_{\text{bits}}$, $\gamma_1 = 0$. |
| Exponent (quanta) | A larger HMM has rarely visited states; if the stationary visitation tail is heavy, $\gamma$ falls with $n$ ($\gamma_1 \lt 0$). Random sparse transition graphs usually have flat-ish stationary distributions, so the quanta picture predicts only a weak exponent shift **unless** the measured tail is heavy. We log the tail and test whether it tracks $\gamma$. |
| Alignment as effective data | $\varepsilon_{\text{learn}} = A(\alpha^\eta T)^{-\gamma}$ with $\eta \approx 1$. |
| Alignment as prior | Part of the alignment effect is $\varepsilon_{\text{id}}(\alpha)$, which no amount of $T$ removes. |
| **Position split (pre-registered)** | Identification happens early in a document, where the training prior matters, so $\alpha$ mainly drives excess loss at positions 1–16. Once the concept is identified, the remaining excess is about modeling the concept's dynamics, so $K$ mainly drives excess loss at positions 65–128. |

The opposite of the position prediction is also plausible. A complex concept is harder to identify (its emission supports overlap more with the distractors'), so $K$ could dominate early too. And with little target data the model may never learn the target's dynamics, so $\alpha$ could dominate late. Gate G5 decides.

## 2. Experimental design

### 2.1 The description language $\mathcal L_{\text{HMM}}$ and the concept generator

A concept is an HMM over a shared vocabulary of $V$ tokens with $n$ hidden states and fixed sparsity:

- **Transitions.** Each state $i$ has exactly 3 distinct successor states $S_i \subset \{0,\dots,n-1\}$ (self-loops allowed), with transition probabilities $k_{ij}/2^q$, where the integers $k_{ij} \ge 1$ sum to $2^q$.
- **Emissions.** Each state $i$ emits from exactly 4 distinct tokens $E_i \subset \{0,\dots,V-1\}$, with probabilities $m_{iv}/2^q$, $m_{iv} \ge 1$, $\sum_v m_{iv} = 2^q$.
- **Start.** The initial state distribution is uniform over the $n$ states. At step $t$ the current state emits $x_t$, then transitions.
- **Precision.** $q = 6$ bits.

**Sampling a candidate concept** (task-sampling seed `20261013`, one NumPy PCG64 stream per arm): draw $S_i$ uniformly without replacement from the $n$ states and $E_i$ uniformly without replacement from the $V$ tokens. Draw the probability vectors from a flat Dirichlet. Quantize them to multiples of $2^{-q}$ with every entry $\ge 2^{-q}$ (largest-remainder rounding after reserving one unit per entry).

**Acceptance.** All checks are in code and their outcomes are logged per candidate:

1. The transition graph is strongly connected and aperiodic (the gcd of cycle lengths is 1), so the entropy rate exists.
2. The process rank equals $n$ (Section 2.2), so no HMM with fewer states generates the same process.
3. **Entropy-rate band.** Larger $n$ could also mean more randomness per token, which would confound complexity with noise. Before any acceptance, each level's first 64 candidates passing checks 1–2 are scored with a quick entropy-rate estimate ($10^5$ tokens). $h_{\text{ref}}$ is the median over all levels pooled. A candidate is accepted only if its quick estimate lies within $\pm 0.15$ bits/token of $h_{\text{ref}}$. If a level cannot supply its 4 concepts within 1,000 candidates, the band widens to $\pm 0.30$ for every level. That is disclosed in `--check-only` output, before the freeze.
4. Accepted concepts are taken in generation order. At each level the first 2 become **targets** and the next 2 become **distractors**. In each arm one more accepted concept per level is a **pilot** concept, used only in the calibration pilot.

### 2.2 Exact complexity and its correctness check

**The exact quantity is the minimal state count $K = n$, certified.** For an HMM write $M_x = D_x A$, where $A$ is the transition matrix and $D_x = \mathrm{diag}(O_{\cdot x})$ holds the emission probabilities of token $x$. Then $P(x_1 \dots x_k) = \pi^\top M_{x_1}\cdots M_{x_k}\mathbf 1$. The Hankel matrix $H[u,v] = P(uv)$ over all strings $u, v$ factors as $H = F B$, with rows $f_u = \pi^\top M_u \in \mathbb R^n$ and columns $b_v = M_v \mathbf 1 \in \mathbb R^n$. So $\operatorname{rank} H \le n$ for **any** $n$-state HMM, and if $\operatorname{rank} H = n$, every HMM (indeed every linear representation) that generates the same process has at least $n$ states. We call $\operatorname{rank} H$ the process rank.

Computation: $\operatorname{rank} H = n$ iff the forward vectors span $\mathbb R^n$ and the backward vectors span $\mathbb R^n$. Each span is computed as a Krylov closure. Start from $\pi^\top$ (respectively $\mathbf 1$), multiply by every $M_x$, keep each new vector that is linearly independent of those kept, and stop when nothing new is added. All entries are integers after scaling by $n \cdot 2^{2q}$ per step. Independence is tested by Gaussian elimination modulo the prime $p = 2^{61}-1$. Integer vectors independent mod $p$ are independent over $\mathbb Q$, so reaching $n$ independent vectors certifies $\operatorname{rank} = n$ exactly. A rank below $n$ mod $p$ (possible by accident) is rechecked with exact `fractions.Fraction` elimination for $n \le 32$. Any candidate not certified at rank $n$ is rejected.

**Description length.** $K_{\text{bits}}$ is the length of a fixed binary encoding in $\mathcal L_{\text{HMM}}$. For each state in index order it stores the 3 successor indices in increasing order ($\lceil\log_2 n\rceil$ bits each), the 4 emission tokens in increasing order ($\lceil\log_2 V\rceil$ bits each), and the first 2 transition numerators and first 3 emission numerators ($q$ bits each, stored as value $-1$; the last numerator in each row is implied by the sum-to-one constraint):

```math
K_{\text{bits}}(n) = n\,\bigl(3\lceil\log_2 n\rceil + 4\lceil\log_2 V\rceil + 5q\bigr).
```

The header ($n$, $V$, $q$) is the same length for every concept and is excluded. $K_{\text{bits}}$ is an **upper bound** on the description length in this language, not a minimum. Relabeling the states gives $n!$ equivalent encodings, so a canonical-labeling code could save up to $\log_2 n!$ bits. It is not universal Kolmogorov complexity. It is monotone in $n$, and $n$ itself is exactly certified. The primary law variable is $K = K_{\text{bits}}$. Two sensitivity analyses repeat every fit with $K = n$ and with $K = K_{\text{bits}} - \log_2 n!$.

**Correctness checks (unit tests).**

- On tiny HMMs ($n \le 4$, $V \le 4$) the Krylov rank equals the exact rank of the brute-force Hankel matrix over all strings of length $\le n-1$ on each side.
- An HMM with two lumpable states (identical emissions and identical outgoing rows) is detected as rank $\lt n$.
- `decode(encode(h)) == h` and `len(encode(h)) == K_bits(n)` for 1,000 random concepts.
- The forward algorithm matches brute-force summation over all hidden paths to $10^{-12}$ on sequences of length 6.

### 2.3 Complexity levels, concept counts and the entropy rate

| Arm | $V$ | Levels $n$ | $K_{\text{bits}}$ ($q = 6$) | Targets per level | Distractors per level | Concepts total |
|---|---|---|---|---|---|---|
| 13-small | 64 | 4, 8, **16**, 32 | 240, 504, **1,056**, 2,208 | 2 | 2 | 16 (+4 pilot) |
| 13-GPU | 150 | 8, 16, **32**, 64, 128 | 568, 1,184, **2,464**, 5,120, 10,624 | 2 | 2 | 20 (+5 pilot) |

Bold = held-out level. $V = 150$ is the largest of GINC's three vocabulary sizes (50, 100, 150). The distractor pool is fixed per arm and shared by every target: $D = 8$ (13-small) or $D = 10$ (13-GPU).

**Entropy rate.** For each concept, $h$ is estimated as the mean of $-\log_2 p(x_t \mid x_{\lt t})$ (forward algorithm) over 100 independent sequences of $10^4$ tokens, discarding positions 1–100 of each, for $\approx 10^6$ tokens in total. Its standard error is the standard deviation of the 100 per-sequence means divided by 10. A unit test checks the estimator on an HMM whose emissions reveal the state (a visible Markov chain), where the exact rate is $-\sum_i \pi_i \sum_j A_{ij}\log_2 A_{ij}$: the estimate must lie within 3 standard errors.

**Mechanism covariates per concept:** the stationary distribution, its rank-frequency tail exponent (least-squares slope of $\log \pi_{(r)}$ on $\log r$), the minimum stationary probability, the expected number of visits to the rarest state per 128-token document, the emission-support overlap with each distractor, and the total Bayes-mixture regret $\varepsilon_{\text{id}}$ at each $\alpha$.

### 2.4 Documents and the alignment construction

- A document is 128 tokens sampled from one concept, preceded by a beginning-of-sequence token (id $V$). The model predicts tokens 1–128. Documents never span two concepts, and there is no concept label.
- **Fresh data, no epochs.** Every training document is new, generated on the fly by NumPy PCG64 seeded by (training seed, concept id, per-concept document index). No document is repeated.
- **$\alpha$ = share of training tokens from the target.** All documents have the same length, so this is also the share of documents. Document $k$ of the stream (0-based) is a target document iff $\lfloor (k+1)\alpha \rfloor - \lfloor k\alpha \rfloor = 1$, so after $m$ documents exactly $\lfloor m\alpha \rfloor$ are target documents, at every checkpoint. Non-target slots cycle through the $D$ distractors in a seeded permutation, repeated, so each distractor's share is $(1-\alpha)/D$ to within one document. Grid: $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$; at $\alpha = 1$ there are no distractors.
- **Common random numbers.** Within a training seed, the $j$-th document of a given concept is identical across all $\alpha$ levels and all targets, and the model initialization is identical across every run of that seed and model size. Conditions differ only in the mixing proportions.

### 2.5 Held-out evaluation set

For each target, 512 held-out documents (65,536 target tokens) in 13-small and 2,048 (262,144 tokens) in 13-GPU, from a separate evaluation stream (seed `20261013 + 7`). `--check-only` hashes every evaluation document and asserts that none equals any training document of any run (training document hashes are recomputed from seeds). Per evaluation token we store the model's, the oracle's and the mixture's code lengths, all computed in float64.

### 2.6 Models, optimizer and checkpoints

**13-small.** Decoder-only transformer, pre-LayerNorm, learned absolute positions (128), 3 layers, $d_{\text{model}} = 96$, 4 heads (head dimension 24), GELU (Gaussian error linear unit) MLP (multilayer perceptron) of width 384, no dropout, tied input/output embedding, vocabulary 65. **Total parameters 354,240**; non-embedding $N = 335{,}712$ (counted from the architecture and checked against a PyTorch instantiation on 10-18-2026; the code must assert both).

**13-GPU.** Same architecture family, vocabulary 151:

| Name | Layers | $d_{\text{model}}$ | Heads | Total parameters | Non-embedding $N$ | Learning rate |
|---|---|---|---|---|---|---|
| S | 4 | 256 | 4 | 3,230,976 | 3,159,552 | $1\times10^{-3}$ |
| M | 8 | 512 | 8 | 25,362,944 | 25,220,096 | $6\times10^{-4}$ |
| L | 12 | 768 | 12 | 85,270,272 | 85,056,000 | $3\times10^{-4}$ |

$N$ in the laws is the non-embedding count ([Kaplan et al. 2020, arXiv 2001.08361](https://arxiv.org/abs/2001.08361)).

**Optimizer (both arms).** AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1 on weight matrices only, gradient clipping at 1.0, linear warmup (32 steps small, 200 steps GPU), then a **constant** learning rate. A constant rate makes every checkpoint a valid point on one run's learning curve. The cost, disclosed here, is that no checkpoint gets a schedule tuned to its own $T$. 13-small trains in float32 with `torch.set_num_threads(1)`. 13-GPU trains with bfloat16 autocast. Evaluation is always float64 with deterministic algorithms.

**Batch and checkpoints.**

- **13-small:** 8 documents per step (1,024 tokens). Checkpoints at $T \in \{2^{17}, 2^{18}, 2^{19}, 2^{20}, 2^{21}\}$ tokens (128 to 2,048 steps). Pre-declared fallback, chosen only from the pilot's measured throughput: if the projected full run exceeds 3.0 h on 4 workers, use $\{2^{16},\dots,2^{20}\}$.
- **13-GPU:** 128 documents per step (16,384 tokens). Checkpoints at $T \in \{2^{22}, 2^{24}, 2^{26}, 2^{28}, 2^{30}\}$, capped per run at 6NT FLOPs $\le 3\times10^{17}$. Under the cap, S and M reach $2^{30}$. L stops at $T_{\max} = 2^{29}$, with checkpoints $\{2^{22}, 2^{24}, 2^{26}, 2^{28}, 2^{29}\}$.

### 2.7 Seeds, grid, fit and held-out cells

- **Seeds.** 13-small: `13101, 13202, 13303, 13404`. That is 4 seeds, not the program's 8: with 8, the full grid would take about 7 h on the 4-core box, over the 4 h budget. 13-GPU: `13101, 13202, 13303` (3 seeds; compute). Pilot seeds `13909` (small) and `13919` (GPU) are never reused. Analysis seed `29101326`. Bootstrap draws 20,000.
- **Grid.** 13-small: 4 levels × 2 targets × 4 $\alpha$ × 4 seeds = **128 runs**. 13-GPU: 5 levels × 2 targets × 4 $\alpha$ × 3 sizes × 3 seeds = **360 runs** (one run = one shard).

| Coordinate | Fit | Held out |
|---|---|---|
| Level $n$ (small) | 4, 8, 32 | 16 |
| Level $n$ (GPU) | 8, 16, 64, 128 | 32 |
| $\alpha$ | 0.05, 0.4, 1.0 | 0.15 |
| $T$ | all but the largest | largest checkpoint of the run |
| $N$ (GPU) | S, M, L | none (three sizes are too few to hold one out and still fit $\beta$ well; a limit, Section 9) |

A row (one run at one checkpoint) is a **fit row** iff none of its coordinates is held out. Every other row is a **held-out row**. Laws are fit per seed on fit rows only. All forecasts for held-out rows are written to `forecasts.csv` and hashed before any held-out outcome is read by the analysis code.

### 2.8 Calibration pilot (disclosed, before the freeze)

- **13-small:** seed `13909`, the 4 pilot concepts, $\alpha \in \{0.05, 0.4\}$, full $T$ to $2^{21}$ (8 runs, 4 workers). It measures training tokens per second per worker under 4-way load (this sets the $T$ range, Section 2.6), checks that no run diverges, and applies the $\varepsilon^\star$ rule of Section 4.4.
- **13-GPU:** seed `13919`, the $n = 32$ pilot concept, $\alpha = 0.4$, learning rate in $\{0.5\times, 1\times, 2\times\}$ the table value for each size, $T$ to $2^{26}$ (9 runs, about 3 A100-hours). The multiplier with the lowest pilot $\varepsilon$ at $2^{26}$ is used for that size; ties within 0.002 bits/token go to $1\times$. This selection rule is part of the frozen design. The GPU pilot necessarily runs after the freeze (the coding agent has no GPU), so its outcome is recorded outside Sections 2–5: in `config.json` key `gpu.lr_selected` and in a "GPU pilot disclosure" paragraph in Section 7, in one commit made before any GPU shard runs.

The 13-small pilot outcome is written into this protocol in the freeze commit. No pilot outcome enters any gate.

**Pilot disclosure (13-small, run 10-18-2026 17:10–17:39 PDT, before the freeze).** Seed `13909`, the 4 pilot concepts ($n = 4, 8, 16, 32$), $\alpha \in \{0.05, 0.4\}$, $T$ to $2^{21}$: 8 of 8 runs finished, none diverged, wall time 1,720 s. Measured under 4-way load (the box also carried a background load of about 3 from other jobs): median training throughput 3,884 tokens/s per worker (range 3,292–4,572), evaluation 37.1 s per checkpoint. Projected grid wall time: 7.50 h for $T_{\max} = 2^{21}$ and 4.58 h for $T_{\max} = 2^{20}$. Decisions, all by the rules fixed above:
- **$T$ range:** the projection exceeds 3.0 h, so the fallback $\{2^{16}, 2^{17}, 2^{18}, 2^{19}, 2^{20}\}$ is used (`small.T_checkpoints`). The largest fit $T$ is $2^{19}$; $2^{20}$ is the held-out checkpoint.
- **$\varepsilon^\star$:** the $n = 32$ pilot concept at $\alpha = 0.4$ had $\varepsilon_{\text{learn}} = 1.093$ bits/token at $2^{19}$, above $2\varepsilon^\star = 0.10$, so the Section 4.4 rule fires and $\varepsilon^\star$ is raised to **1.10 bits/token** (`small.epsilon_star`). This is far above the 0.05 the design expected: on this box the 354k-parameter model learns these concepts slowly (at $2^{20}$ the pilot $\varepsilon_{\text{learn}}$ was 0.17, 0.20, 0.40, 0.90 bits for $n = 4, 8, 16, 32$ at $\alpha = 0.4$, and 0.43, 0.52, 1.10, 1.99 at $\alpha = 0.05$). The G4 model-free fallback level (Section 5 alignment note) is therefore $2\varepsilon^\star = 2.2$ bits.
- **Evaluation noise:** median standard error of $\varepsilon$ over all positions 0.0048 bits/token (max 0.0095), in line with the 0.004 assumed in Section 4.4; $\varepsilon^\star$ is more than 100 times it.
- **Resolution floor** (needed by the Experiment 11 censored residual this protocol inherits; not set elsewhere here): $\varepsilon_{\text{res}} = 0.015$ bits/token, about 3 times the median pilot standard error (`shared.eps_res`). No pilot cell came near it (smallest 0.14).
- **Wall cap (deviation, decided before the freeze):** even the fallback projects 4.58 h, over the 4 h cap of Section 7, because measured throughput was about 35% below the 6,000 tokens/s assumed and evaluation about 7 times slower. Rather than leave about 16 runs of the fourth seed unfinished (which would unbalance the Section 4.5 design), `small.max_wall_seconds` is raised from 14,400 to 18,000 s (5 h). No gate, law, threshold or seed changes.
- Seed `13909` is never used again.

## 3. Measurements

Per run and checkpoint, on the target's held-out documents:

- **Primary:** $\varepsilon = L_{\text{model}} - L^\star$, in bits/token, averaged over all 128 positions.
- **Slices:** the same average over **early** positions 1–16 (identification), **middle** 17–64, and **late** 65–128 (modeling), plus the per-position curve for all 128 positions.
- **References:** $\varepsilon_{\text{id}}(\alpha) = L_{\text{mix},\alpha} - L^\star$ and $\varepsilon_{\text{learn}} = \varepsilon - \varepsilon_{\text{id}}$, per slice. The mixture uses the run's own training prior ($\alpha$, $D$).
- **Posterior check:** the model's implied concept identification is not observable directly. We record the mixture's posterior on the target after 16 tokens, averaged over held-out documents, as the identification difficulty of each (target, $\alpha$).
- **Training-distribution loss:** the model's loss on 64 fresh documents from each distractor. Not gated; it shows capacity sharing.
- **Raw loss** $L_{\text{model}}$ and oracle loss $L^\star$ per slice; tokens counted, wall time, and tokens per second.
- **Diversity link (not gated).** If Experiment 14's code exists (`experiments/14_task2vec_vs_exact_complexity/expt_v1/` or its archive copy), compute the Task2Vec diversity coefficient of each (target, $\alpha$) training mix with Experiment 14's diversity pipeline and settings, on 13-small only, and log it to `diversity.csv`. Experiment 14 predicts that the coefficient tracks the mixture's concept entropy, $H(\alpha) = -\alpha\log_2\alpha - (1-\alpha)\log_2\frac{1-\alpha}{D}$, which falls as $\alpha$ rises, and is nearly flat in $K$. If the code does not exist, write "follow-up: diversity coefficient of the Experiment 13 mixes" in the results file.

## 4. Laws and analysis plan

### 4.1 Laws (fit per seed on fit rows, on $y = \log_2 \varepsilon$ over all positions)

Every effective-data form carries the exact, non-fitted identification offset $\varepsilon_{\text{id}}(\text{target}, \alpha)$ of the row. The overview's forms assume $\varepsilon \to 0$ as $T \to \infty$, but here $\varepsilon$ approaches $\varepsilon_{\text{id}}(\alpha) \gt 0$ whenever $\alpha \lt 1$. Fitting a pure power law to $\varepsilon$ would be misspecified, and the misspecification would look like an $\alpha$ effect. Write $\tilde K = K/K_{\text{ref}}$ (geometric mean of the fit levels' $K$), $\tilde T = T/T_0$ (smallest checkpoint), and $\gamma(K) = \gamma_0 + \gamma_1\log_2\tilde K$:

| Name | Forecast $\hat\varepsilon$ |
|---|---|
| `M0_scale` | $\varepsilon_{\text{id}} + A\,\tilde T^{-\gamma}$ |
| `M_align_eff` | $\varepsilon_{\text{id}} + A\,(\alpha^\eta \tilde T)^{-\gamma}$ |
| `M_align_add` | $a + c\,\tilde T^{-\gamma} - d\,\alpha$ (the alignment-scaling paper's first-order form, applied to $\varepsilon$; no exact offset) |
| `M_K_coef` | $\varepsilon_{\text{id}} + A\,\tilde K^{\kappa}(\alpha^\eta \tilde T)^{-\gamma}$ |
| `M_K_exp` | $\varepsilon_{\text{id}} + A\,(\alpha^\eta \tilde T)^{-\gamma(K)}$ |
| `M_K_both` | $\varepsilon_{\text{id}} + A\,\tilde K^{\kappa}(\alpha^\eta \tilde T)^{-\gamma(K)}$ |

**13-GPU full laws** (pooled over $N$, also per seed): the effective-data form $\hat\varepsilon = \varepsilon_{\text{id}} + B\,\tilde N^{-\beta} + A\,\tilde K^{\kappa}(\alpha^\eta\tilde T)^{-\gamma(K)}$, and the alignment-scaling form $\hat\varepsilon = a + b\,\tilde N^{-\beta} + c_0\tilde K^{\kappa}\,\tilde T^{-\gamma(K)} - d\,\alpha$, with $\tilde N = N/3\times10^6$. Also fit: the same two forms with $\kappa = 0$, and with $\gamma_1 = 0$.

**Fitting.** Nonlinear least squares on $\log_2\varepsilon$ (`scipy.optimize.least_squares`, trust-region reflective), 64-start grid over the exponents, bounds $\gamma, \gamma_0 \in (0, 3]$, $\gamma_1 \in [-1, 1]$ with $\gamma(K) \gt 0$ enforced at every grid $K$, $\eta \in [0, 3]$, $\kappa \in [-3, 3]$, $\beta \in (0, 3]$, $A, B \gt 0$. `M_align_add` is fit in linear space with the same starts. A forecast is **invalid** if it is nonfinite or below the floor ($\hat\varepsilon \le 0$, or $\hat\varepsilon \lt \varepsilon_{\text{id}}$ for forms without the offset). `independent_fit.py` refits every law with a different algorithm (`scipy.optimize.minimize`, L-BFGS-B on the same objective, different starting grid) and must agree with the primary fit's held-out root-mean-square error (RMSE) to $10^{-4}$ relative before anything is reported. **Sensitivity:** every fit is repeated without the exact offset, and with $K = n$ and $K = K_{\text{bits}} - \log_2 n!$. These are reported, not gated.

### 4.2 Held-out scoring

For each seed and law: RMSE of $\log_2\hat\varepsilon - \log_2\varepsilon$ over held-out rows (invalid forecasts are scored at $\log_2$ of the floor plus $10^{-6}$ and also counted). Comparison: relative RMSE reduction $1 - \mathrm{RMSE}_{\text{new}}/\mathrm{RMSE}_{\text{base}}$, computed on the seed-averaged squared errors. Also reported by held-out coordinate (level only, $\alpha$ only, $T$ only).

### 4.3 Uncertainty

95% percentile intervals from a paired bootstrap over seed blocks: resample seeds with replacement, 20,000 draws, analysis seed `29101326`. Rows and checkpoints from one seed are never treated as independent. With 4 seeds (small) the bootstrap has only 35 distinct resamples, and with 3 seeds (GPU) only 10. So in 13-GPU every gate also requires each seed's own point estimate to fall on the supported side. A secondary two-stage bootstrap resamples seeds and then the 2 targets within each level, to show the size of task-sampling variance. It is reported, not gated.

### 4.4 Data multipliers and the target excess $\varepsilon^\star$

$T^\star(K, \alpha)$ is the $T$ at which the fitted learning term equals $\varepsilon^\star$. The exact identification offset is excluded, so the multipliers compare data bills, not floors:

```math
T^\star(K,\alpha) = T_0\,\alpha^{-\eta}\Bigl(A\tilde K^{\kappa}/\varepsilon^\star\Bigr)^{1/\gamma(K)} .
```

- **Complexity multiplier:** $M_K = T^\star(K_{\max}, \alpha)/T^\star(K_{\min}, \alpha)$ over the fit levels. It is independent of $\alpha$.
- **Alignment range multiplier:** $M_\alpha = T^\star(K, 0.05)/T^\star(K, 1) = 20^{\eta}$.
- **Elasticities:** $e_\alpha = \eta$ and $e_K = \partial\log T^\star/\partial\log K$ at $K_{\text{ref}}$.

All are computed per seed from `M_K_both`. If `M_K_both` fails to converge for more than 25% of seeds, `M_K_coef` is used instead, and the switch is reported. A model-free check interpolates each cell's observed $\varepsilon_{\text{learn}}$ curve log-log between checkpoints. It is reported, not gated.

**$\varepsilon^\star$.** Default $\varepsilon^\star = 0.05$ bits/token for 13-small and $0.02$ bits/token for 13-GPU, applied to $\varepsilon_{\text{learn}}$ over all positions. Why these values:

1. They are about 2% (small) and about 1% (GPU) of the entropy-rate band's center (expected near 2 bits/token), so reaching them means the concept is modeled almost to the floor.
2. They are at least 10 times the evaluation standard error of $\varepsilon$, which is about 0.004 bits/token at 65,536 tokens and 0.002 at 262,144 tokens (the pilot measures this and must confirm it).
3. They sit inside the fitted range rather than beyond it.

Pilot rule (13-small only, applied before the freeze): if the $n = 32$ pilot concept at $\alpha = 0.4$ does not reach $\varepsilon_{\text{learn}} \le 2\varepsilon^\star$ by the largest fit $T$, $\varepsilon^\star$ is raised to its $\varepsilon_{\text{learn}}$ at that $T$, rounded up to the next 0.01. The 13-GPU value 0.02 is fixed at the freeze and has no pilot rule. $T^\star$ comes from the fitted law, so a level that has not reached $\varepsilon^\star$ by its last checkpoint gives an extrapolated $T^\star$, which is flagged in the report. Multiplier curves over $\varepsilon \in \{0.01, 0.02, 0.05, 0.1, 0.2\}$ are reported regardless.

### 4.5 Position decomposition (for G5)

For each seed and each slice $s \in \{\text{early}, \text{late}\}$, take $y = \log_2\varepsilon_s$ on the full balanced grid of that arm, averaged over the 2 targets within a level. That is 4 levels × 4 $\alpha$ × 5 $T$ = 80 cells for 13-small, and per size for 13-GPU. Fit the additive main-effects model $y = \mu + a_\alpha + b_K + c_T + r$ (all three factors categorical). Because the design is balanced, the main-effect sums of squares $SS_\alpha$, $SS_K$, $SS_T$ are orthogonal. Define the **alignment share** $r_\alpha(s) = SS_\alpha/(SS_\alpha + SS_K)$. The prediction is $r_\alpha(\text{early}) \gt 0.5 \gt r_\alpha(\text{late})$. We also report the same shares for $\varepsilon_{\text{learn}}$ and $\varepsilon_{\text{id}}$ separately, and for the middle slice.

### 4.6 Mechanism analyses (reported, not gated)

- Per target, fit $\gamma$ at $\alpha = 1$. Spearman correlation across targets between that $\gamma$ and the stationary tail exponent, which tests the quanta mechanism.
- $\varepsilon_{\text{id}}(\alpha)$ against $\alpha$: is it linear (the alignment-scaling paper's $-d\alpha$) or closer to $\log(1/\alpha)$? Compare the residual sums of squares of the two one-parameter-per-target fits.
- The share of early-slice excess that is $\varepsilon_{\text{id}}$ at the largest $T$.
- Distractor loss against target loss (capacity sharing).

## 5. Prospective pass/fail gates

Every gate ends `SUPPORTED` or `NOT_ESTABLISHED` and is reported with equal prominence. Intervals are those of Section 4.3. G1–G5 are evaluated separately in 13-small and in 13-GPU (in 13-GPU on the full $N$ law, pooled, and per size). G1–G5 are primary. G6 (13-GPU only) is secondary. These mirror the Experiment 11 gate family. If Experiment 11's frozen protocol defines G1–G4 differently, the running agent aligns the definitions **before** this protocol's freeze and records the change in the freeze commit.

**Alignment with Experiment 11 (made at the freeze, 13-small).** Experiment 11's frozen G1–G4 differ from the drafts below, so for 13-small the gated G1–G4 are Experiment 11's, with this experiment's $\varepsilon^\star$ (1.10 bits, pilot), $K$ in description-length bits (`K_bits`) and held-out level $n = 16$:
- **G1:** $M_K$ at $\alpha = 1$ from model-free $T^\star$ (log-log interpolation of each run's $\varepsilon_{\text{learn}}$ curve), geometric mean over targets of $n = 32$ over $n = 4$. `SUPPORTED` iff $M_K \ge 2.0$, seed-block 95% lower bound $\gt 1$ and two-stage (seeds and targets) 95% lower bound $\gt 1$. Censored $T^\star$ (never reaching, or already below, $\varepsilon^\star$) is set to the nearest checkpoint and counted in the report.
- **G2:** the $K$ law is whichever of `M_K_coef` and `M_K_exp` has the lower pooled fit-row RMSE, chosen before held-out rows are read. `SUPPORTED` iff its held-out RMSE reduction against `M_align_eff` is $\ge 10\%$ with 95% lower bound $\gt 0\%$, its RMSE is also lower on the held-out level's rows alone, and it has 0 invalid forecasts.
- **G3a (coefficient):** in `M_K_both`, $\kappa$'s 95% interval above 0, $\gamma_1$'s interval contains 0, and held-out RMSE of `M_K_coef` $\le$ `M_K_exp`. **G3b (exponent):** $\gamma_1$'s 95% interval below 0, and `M_K_exp` cuts held-out RMSE relative to `M_K_coef` by $\ge 5\%$ with 95% lower bound $\gt 0\%$. Neither: C3 unresolved.
- **G4:** the elasticity ratio $E = e_\alpha / e_K$ from `M_K_both` (refit on all rows, descriptive) at $\varepsilon^\star$ is $\gt 1$ with seed-block 95% lower bound $\gt 1$, in both $K$ units (bits and states), and the model-free $E_{\text{mf}}$ point estimate is $\gt 1$; if more than half of the (target, $\alpha$) pairs are censored at $\varepsilon^\star$ in the median seed, $E_{\text{mf}}$ is evaluated at $2\varepsilon^\star$.
- **G5:** this protocol's position split, unchanged.

The draft versions below (G2 Bonferroni pair, G3 ratio $\rho$, G4 range ratio $R$ and gap ratio $\Delta$, and the law-based $M_K$) are computed and reported next to the gates, not gated. G6 (13-GPU) is unchanged; 13-GPU is not run in this freeze.

1. **G1 — Complexity is real (C1).** `SUPPORTED` iff $M_K \gt 1$ with 95% lower bound $\gt 1$ at $\varepsilon^\star$.
2. **G2 — Complexity adds held-out value beyond alignment.** Two comparisons: `M_K_coef` vs `M_align_eff` and `M_K_exp` vs `M_align_eff`, on held-out rows. `SUPPORTED` iff at least one gives a relative RMSE reduction $\ge 10\%$ with a Bonferroni-corrected lower bound $\gt 0\%$ (2.5th percentile replaced by the 1.25th), and the winning law has 0 invalid forecasts. It must also improve on the held-out level's rows alone (relative reduction $\gt 0$).
3. **G3 — Coefficient or exponent (C3).** Let $\rho = \mathrm{RMSE}(\texttt{M\_K\_exp})/\mathrm{RMSE}(\texttt{M\_K\_coef})$ on held-out rows. **Exponent** `SUPPORTED` iff $\rho \le 0.90$ with 95% upper bound $\lt 1$, **and** $\gamma_1 \lt 0$ in `M_K_both` with 95% upper bound $\lt 0$. **Coefficient** `SUPPORTED` iff $1/\rho \le 0.90$ with 95% upper bound $\lt 1$, **and** the 95% interval of $\gamma_1$ in `M_K_both` contains 0 or lies within $[-0.02, 0.02]$. Otherwise G3 is `NOT_ESTABLISHED` (undecided). The report states which picture, if either, was supported. If the exponent picture wins, it also states $\varepsilon_{\text{flip}}$, the excess at which $M_K(\varepsilon)$ would equal $M_\alpha$.
4. **G4 — Alignment dominates once trained (C2).** Two parts, both required. (a) The range ratio $R = \log M_\alpha / \log M_K$ at $\varepsilon^\star$ is $\gt 1$ with 95% lower bound $\gt 1$: across the grid, alignment moves the data bill more than complexity does. (b) At $\alpha = 1$, the complexity gap $\Delta(T) = \varepsilon(K_{\max}) - \varepsilon(K_{\min})$ (observed, averaged over targets) satisfies $\Delta(T_{\text{last fit}})/\Delta(T_{\text{first}}) \lt 1$ with 95% upper bound $\lt 1$. The elasticity ratio $e_\alpha/e_K$ is reported alongside.
5. **G5 — Alignment acts early, complexity acts late.** `SUPPORTED` iff $r_\alpha(\text{early})$ has 95% lower bound $\gt 0.5$ **and** $r_\alpha(\text{late})$ has 95% upper bound $\lt 0.5$. The two one-sided failures are named in the report: "$\alpha$ dominates everywhere" ($r_\alpha(\text{late})$ lower bound $\gt 0.5$) and "$K$ dominates everywhere" ($r_\alpha(\text{early})$ upper bound $\lt 0.5$).
6. **G6 (13-GPU, secondary) — Does a bigger model shrink the complexity multiplier?** Fit `M_K_both` separately per size and compute $M_K(N)$ at $\varepsilon^\star$. `SUPPORTED` iff $M_K(\text{L})/M_K(\text{S}) \le 0.67$ with 95% upper bound $\lt 1$, and the point estimates are monotone ($M_K(\text{S}) \ge M_K(\text{M}) \ge M_K(\text{L})$). The reverse outcome (ratio $\ge 1.5$ with lower bound $\gt 1$: bigger models pay relatively more for complexity) is reported by name.

## 6. What each outcome would mean

| Outcome | Interpretation |
|---|---|
| G1, G2, G4 `SUPPORTED` | Brando's claim holds on GINC-style data: complexity costs data, and alignment's effect on the data bill is larger across this grid. |
| G1 `SUPPORTED`, G4 (a) fails | Complexity rivals or beats alignment here. With $\varepsilon_{\text{id}}$ removed, alignment may matter mostly through the prior (identification), not through effective data. Check $\eta$. |
| G3 coefficient | Description length enters like an MDL parameter count: a constant data multiplier. Matches the MDL prediction. |
| G3 exponent | Complexity changes the learning rate itself. If the stationary-tail correlation is also strong, this is evidence for the quanta mechanism. If not, the mechanism is something else and must be named as unknown. |
| G2 `NOT_ESTABLISHED`, G1 `SUPPORTED` | Complexity matters, but `M_align_eff` with the exact offset already forecasts held-out cells well, so the $K$ term does not generalize to the held-out level. |
| G5 `SUPPORTED` | Clean mechanistic split: alignment pays for identification, complexity for modeling. It also explains why benchmark "difficulty" can look like alignment in short contexts and like complexity in long ones. |
| G5 "$K$ dominates everywhere" | Complex concepts are also harder to identify (emission overlap). Check the overlap covariate. Alignment then has no special early role. |
| G6 `SUPPORTED` / reverse | Scale buys back complexity cost, or makes it relatively worse. Either way, a $K \times N$ term belongs in the full law. |
| $\varepsilon_{\text{id}}$ fits $\log(1/\alpha)$ far better than linear | The $-d\alpha$ term of the alignment-scaling law is a first-order approximation to an identification cost. That is a concrete suggestion for the paper with Varun. |
| Confound: the entropy-rate band had to widen | The levels differ in noise as well as in $n$. Results hold for $n$ together with $h$, and the report says so. |
| Confound: the pilot moved $\varepsilon^\star$ or the $T$ range | Disclosed. The gates use the frozen values. |

## 7. Compute estimate and hardware

**13-small (CPU, 4 cores, 15 GB RAM, no GPU).** Measured on 10-18-2026 on the 4-core box itself, PyTorch 2.14 CPU build, 1 thread per process: the 354,240-parameter model trained at about 10,500–11,000 tokens/s with batches of 8–16 documents, and about 6,500 tokens/s per process with 4 processes of 32-document batches running at once. Assuming 6,000 tokens/s per worker under 4-way load:

- $T_{\max} = 2^{21}$: about 350 s of training + about 25 s for evaluation (5 checkpoints × 65,536 tokens in float64) + about 10 s of sampling ≈ 385 s per run. 128 runs on 4 workers ≈ **3.4 h**.
- Fallback $T_{\max} = 2^{20}$: ≈ 200 s per run, ≈ **1.8 h**.
- Precomputation (concept acceptance, process ranks, entropy rates at $10^6$ tokens for 20 concepts, oracle and mixture code lengths for 8 targets × 4 $\alpha$): under 15 min.
- Pilot: 8 runs ≈ 15 min.

The pilot decides between $2^{21}$ and $2^{20}$ (Section 2.6). Hard cap in `config.json`: `max_wall_seconds = 14400` (4 h). RAM: about 0.4 GB per worker. Artifacts about 1.0 GB (640 checkpoints × 1.4 MB, per-position tables, evaluation documents).

**13-GPU.** Training FLOPs ≈ $6NT$ (attention adds about 10% at context 128).

| Size | $T_{\max}$ | FLOPs per run | Assumed sustained throughput (A100-80GB, bf16) | Hours per run (A100) | Runs | A100-hours |
|---|---|---|---|---|---|---|
| S | $2^{30}$ | $2.0\times10^{16}$ | about 20 TFLOP/s (small, launch-overhead bound) | 0.3 | 120 | 36 |
| M | $2^{30}$ | $1.6\times10^{17}$ | about 60 TFLOP/s | 0.8 | 120 | 96 |
| L | $2^{29}$ | $2.7\times10^{17}$ | about 100 TFLOP/s | 0.8 | 120 | 96 |
| Evaluation, data generation, pilot | | | | | | about 25 |
| **Total** | | | | | **360** | **about 250** |

H200 (skampere2) or B200 (skampere3) should be roughly 1.5–2.5 times faster. One 8-GPU node therefore needs about 30 h (A100) or 15–20 h (H200), and Sherlock's 16-GPU per-user cap gives about 16 h if GPUs are free. Training data is generated by 8 CPU workers per GPU (NumPy PCG64, so the stream does not depend on GPU type). GPU memory: L at 16,384 tokens per step needs about 8 GB, so every listed GPU fits, including mercury's 16 GB RTX A4000. The A4000 is about 4 times slower than an A100, so use it only for S and M shards.

**Shard plan.** One shard = one run. Shard id $= 120\,i_N + 24\,i_n + 12\,i_{\text{target}} + 3\,i_\alpha + i_{\text{seed}}$ with $i_N \in \{0,1,2\}$ for S, M, L. So shards 0–119 are S, 120–239 are M, and 240–359 are L. A shard writes `shards/<id>/` and finally a `DONE` marker holding the SHA-256 of its files. Launchers claim a shard by atomic `mkdir claims/<id>` and skip shards with `DONE`, so a relaunch resumes without duplicating work. A claimed shard without `DONE` is a crash. It is moved aside, logged in the CKPT file and relaunched by hand, never automatically. Suggested split: skampere2 (8×H200) shards 0–239 and skampere1 (8×A100) shards 240–359, or one Sherlock array `0-359%16`. Storage: float32 checkpoints total about 275 GB (L alone about 205 GB). They stay on cluster storage (SNAP DFS or Sherlock `$SCRATCH`, copied to `$GROUP_HOME` or Oak before the 90-day purge) and are never committed; the repository receives metrics, forecasts, manifests and every checkpoint's SHA-256. Launch scripts are in [`launch/`](launch/README.md).

## 8. Artifacts and verification

Under `expt_v1/results/small/` and `expt_v1/results/gpu/`, following the program's shared output contract adapted to sequence data:

- `tasks.json`: every concept with `task_id`, `family: "hmm"`, `role` (target, distractor or pilot), `K` (= $n$, certified), `K_bits`, `definition` (successor lists, emission lists, integer numerators, $q$, $V$), `process_rank`, entropy rate with standard error, stationary distribution, `visit_tail_exponent`, minimum stationary probability, the acceptance log (candidates tried, band used).
- `data/<task_id>_test.npz`: the held-out documents (int16 token arrays). Training documents are not stored. They are regenerated from seeds, and `mixes.json` records each run's per-concept document counts at every checkpoint, the data-order seed, and a SHA-256 over the training stream.
- `oracle.npz` and `mixture.npz`: per-token oracle and mixture code lengths (float64) per (target, $\alpha$).
- `metrics.csv`: `seed, target_id, K, K_bits, alpha, N, T, step, slice, excess_bits, ident_bits, learn_bits, model_bits, oracle_bits, mixture_bits, n_tokens`. `per_position.csv`: the 128-position curves.
- `checkpoints/<seed>_<target_id>_<alpha>[_<size>]/T_<T>.npz`: float32 weights as NumPy arrays, so replay needs no pickle.
- `fits.json`, `forecasts.csv` (hashed before scoring), `gates.json`, `bootstrap.npz`, `diversity.csv` (if computed), `manifest.json` (source SHA-256s, versions, hardware, wall time, pilot outcomes), and `ARTIFACT_SHA256.json` at the end.

`verify.py` must print PASS only if all of these hold:

1. It reloads every checkpoint and recomputes the held-out model code lengths in float64 to `max_abs_diff < 1e-9` bits per slice.
2. It recomputes oracle and mixture code lengths with an independent log-space forward algorithm (not the scaled forward used in `run.py`) to $10^{-9}$.
3. It recertifies every process rank and recomputes every $K_{\text{bits}}$ from the stored definitions.
4. It regenerates every run's training-stream hash from seeds.
5. It refits every law, and the refit agrees with `fits.json` to $10^{-4}$ relative.

13-GPU verification runs on GPU nodes through the same launch scripts (`MODE=verify`).

## 9. Scope and limits

- $K$ here is the minimal state count and a fixed-encoding description length in $\mathcal L_{\text{HMM}}$. Neither is universal Kolmogorov complexity, and results are stated for this language. A different HMM parameterization (dense transitions, as in GINC) would give different $K_{\text{bits}}$.
- Two targets per level is a small sample of tasks. Task-sampling variance is shown by the secondary bootstrap but not gated.
- 4 seeds (small) and 3 seeds (GPU) make the seed-block intervals coarse.
- The constant learning rate favors early checkpoints relative to a tuned schedule.
- Excess loss depends on the 128-token document length. Longer documents would shrink the per-token identification cost.
- The entropy-rate band controls noise per token but not emission overlap or the identifiability of the target among the distractors, which grow with $n$ by construction. They are measured covariates, not controls.
- No held-out model size, so the $N$ terms are fit, not extrapolated.
- The results apply to these learners, budgets and toys. They do not show that benchmark difficulty in large language models is mostly alignment.

**TLDR-end:** [tdc: Expt 13 design] Train small decoder transformers on GINC-style mixtures of sparse hidden Markov model concepts whose minimal state count is certified exactly, crossing target complexity ($n$ = 4–32 on CPU, 8–128 on GPU) with target share $\alpha$ (0.05–1) and, on GPU, model size, then score excess loss against the exact oracle and Bayes-mixture floors to test whether complexity costs data, whether alignment dominates, whether complexity moves the coefficient or the exponent, and whether alignment acts early in a document and complexity late.
