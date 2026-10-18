# Protocol: Exact shortest-program complexity versus data alignment on bit-word functions (`12_dsl_shortest_program_complexity_vs_alignment`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/12_dsl_shortest_program_complexity_vs_alignment/PROTOCOL.md>

Status: DRAFT design, 10-18-2026, from Brando Miranda's complexity-versus-alignment idea. The running agent may revise it, then must freeze it in its own commit before any gated run; after the freeze nothing in Sections 2–5 changes.

Notation follows the [program overview](../ideas/complexity_vs_alignment/README.md): $\varepsilon$ is held-out excess loss in bits, $T$ is training examples seen, $\alpha$ is the target's share of the training mix, $K$ is exact complexity in a stated description language, and the laws are `M0_scale`, `M_align_eff`, `M_align_add`, `M_K_coef`, `M_K_exp` and `M_K_both` with parameters $A, \eta, \kappa, \gamma, \gamma_0, \gamma_1$. The design mirrors [Experiment 11](../11_dfa_exact_complexity_vs_alignment/PROTOCOL.md) and changes only what the DSL family requires.

## 1. Scientific motivation

**Hypothesis.** Brando (10-18-2026): "kolmogorov complexity would mean that it does take more 'effort' to train, but if you already trained on it the models will perform better. So it's not that complexity doesn't exist but data alignment dominates." Experiment 11 tests this with automata. The overview asks for a second, independent exact-$K$ language, so that a result is not an artifact of automata. Brando also asked for "experiments where we can actually compute the kolmogorov complexity or something really close to it? like what sipser's book has". Sipser (*Introduction to the Theory of Computation*, Section 6.4) defines descriptive complexity as the length of the shortest description under a fixed description method. A fixed, small programming language makes that length computable exactly.

**The description language $\mathcal L_{\text{DSL}}$.** Objects are functions $f: \{0,1\}^{10} \to \{0,1\}^{10}$ on 10-bit words. Bit $i$ of a word $x$ is $(x \gg i)\,\&\,1$, so bit 0 is the least significant. The language has 12 primitives:

| Primitive | Definition on $x \in [0, 1023]$ | Primitive | Definition |
|---|---|---|---|
| `NOT` | $x \oplus 1023$ | `GRAY` | $x \oplus (x \gg 1)$ |
| `REV` | bit $i$ moves to bit $9-i$ | `PREFIX_XOR` | $y = x$; then $y \mathrel{\oplus}= y \gg s$ for $s = 1, 2, 4, 8$ (inverse of `GRAY`) |
| `ROTL1` | $((x \ll 1) \mid (x \gg 9))\,\&\,1023$ | `SWAP_HALVES` | $((x\,\&\,31) \ll 5) \mid (x \gg 5)$ |
| `ROTR1` | $((x \gg 1) \mid (x \ll 9))\,\&\,1023$ | `SWAP_ADJ` | $((x\,\&\,341) \ll 1) \mid ((x \gg 1)\,\&\,341)$ (swap bit pairs 0–1, 2–3, …) |
| `SHL1` | $(x \ll 1)\,\&\,1023$ | `INC` | $(x + 1) \bmod 1024$ |
| `SHR1` | $x \gg 1$ | `NEG` | $(-x) \bmod 1024$ |

A program $p = (p_1, \dots, p_L)$ computes $x \mapsto p_L(\cdots p_1(x))$, and the empty program is the identity. The exact complexity of $f$ is

```math
K(f) = \min\{L : \text{some program of length } L \text{ has the same 1{,}024-entry truth table as } f\},
```

found by breadth-first search over **distinct functions**. Each new function is deduplicated by exact truth-table comparison against every function already seen (Section 2.2). To turn length into bits, a program is written as the Elias gamma code of $L$ ($2\lfloor\log_2 L\rfloor + 1$ bits) followed by the index of the program among the $12^L$ sequences of length $L$ ($\lceil L\log_2 12\rceil$ bits):

```math
K_{\text{bits}}(K) = 2\lfloor \log_2 K\rfloor + 1 + \lceil K \log_2 12 \rceil, \qquad K_{\text{bits}} = 11,\ 14,\ 20,\ 23,\ 27 \ \text{ for } K = 2, 3, 4, 5, 6 .
```

The code is prefix-free, and $K_{\text{bits}}$ increases with $K$, so it is the exact shortest description of $f$ in this bit language. An interpreter for $\mathcal L_{\text{DSL}}$ that prints the truth table gives $K(\text{truth table of } f) \le K_{\text{bits}}(f) + c_{\mathcal L}$ for prefix Kolmogorov complexity. The bound can be very loose, and $K_{\text{bits}}$ is not universal Kolmogorov complexity. Every claim below is stated for this language only.

**Claims tested** (from the overview):

- **C1, complexity is real.** Functions with longer shortest programs cost more data to reach the same held-out loss.
- **C2, alignment dominates once trained.** In units of data, changing the target share moves the bill more than changing $K$ does, and at $\alpha = 1$ the complexity gap in loss shrinks as $T$ grows.
- **C3, which law.** Does complexity move the coefficient (minimum description length (MDL) picture, `M_K_coef`: $\kappa \gt 0$, $\gamma_1 = 0$) or the exponent (quanta picture, `M_K_exp`: $\gamma_1 \lt 0$)?

**Competing predictions.**

| Picture | Prediction here |
|---|---|
| MDL / Occam (Grünwald 2007, ch. 8; [Grünwald's MDL tutorial](https://arxiv.org/abs/math/0406077)) | Excess loss scales like $K_{\text{bits}}/T$: $\kappa \gt 0$, $\gamma_1 = 0$. The complexity data multiplier is the same at every accuracy. |
| Quanta ([Hutter 2021](https://arxiv.org/abs/2102.04074); [Michaud et al. 2023](https://arxiv.org/abs/2303.13506)) | Longer programs that use `INC` or `NEG` split the inputs into more **local rules** (Section 2.3), some of them rare, such as long carry chains. A heavier tail of rare rules lowers the exponent: $\gamma_1 \lt 0$. Gate G7 checks the mechanism directly. |
| Sensitivity and parity ([Hahn and Rofin 2024](https://arxiv.org/abs/2402.09963); [Barak et al. 2022](https://arxiv.org/abs/2207.08799)) | Transformers find high-sensitivity functions hard, and parities of high degree are hard for gradient descent. Here every program without `INC` or `NEG` is affine over GF(2), so each output bit is a parity of input bits. Both sensitivity and the affine share change with $K$ (Section 2.4), so a $K$ effect could be either of them. Both are controlled by design and regressed out. |
| Simplicity bias toward compressible truth tables ([Valle-Pérez, Camargo and Louis 2018](https://arxiv.org/abs/1805.08522); [Bhattamishra et al. 2023](https://arxiv.org/abs/2211.12316)) | Learnability follows how compressible the truth table is, not program length. A prototype found that compressed size *falls* with $K$ here (Section 3.2), so this picture and C1 predict opposite signs for learning cost. |
| Data repetition ([Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)) | Each task has only 512 training inputs, so $T$ counts repeated passes. Repeats are worth less than fresh data, which could flatten the $\alpha = 1$ curves and bias $\eta$. A secondary arm with 768 training inputs measures this (Section 2.6). |

**Why the alignment variable is clean here.** All tasks share the same 512 training inputs and the same 512 test inputs, so a target example and a distractor example differ only in the output and in the task token. Without the task token, the mix would be a noisy-label problem whose loss floor depends on $\alpha$ (Experiment 13 studies that case). With it, labels from different tasks never conflict, the floor is 0 at every $\alpha$, and $\alpha$ acts only through the data share and through shared parameters. The token makes identification trivial. That is intended: it isolates alignment as data share. Every example has exactly 11 tokens, so the token-weighted alignment of [the alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws) (Eq. 2, with $s(x) = 1$ for target examples and $0$ otherwise) equals the example share $\alpha$ exactly. Distractors are compositions of the same primitives, so they can transfer (shared sub-programs, $\eta \lt 1$) or interfere (competition for capacity, $\eta \gt 1$).

**Why CTM and BDM.** The Coding Theorem Method (CTM) estimates the complexity of short strings from how often small Turing machines output them ([Soler-Toscano, Zenil, Delahaye and Gauvrit, PLoS ONE 2014, arXiv 1211.1302](https://arxiv.org/abs/1211.1302)). The Block Decomposition Method (BDM) extends CTM to long strings by summing CTM values of short blocks plus a log-multiplicity term ([Zenil, Hernández-Orozco, Kiani, Soler-Toscano and Rueda-Toicen, Entropy 2018, arXiv 1609.00110](https://arxiv.org/abs/1609.00110)). These are the standard computable stand-ins for Kolmogorov complexity. Exact $K$ in a fixed language is a third. The three are compared on the same 200 functions, and each is tested against learning cost.

## 2. Experimental design

### 2.1 Inputs, splits and outputs

- **Input space.** All 1,024 10-bit words. Inputs are drawn uniformly from a partition, so the input distribution is identical for every task (total-variation distance 0).
- **Train/test partition, shared by every task.** A fixed permutation $\pi$ of $0..1023$ (task seed 20261012) gives train inputs $\pi[0{:}512]$ and test inputs $\pi[512{:}1024]$. The permutation is redrawn (logged) until every input bit is 1 in between 236 and 276 of the 512 train inputs. No test input is ever seen in training under any task token.
- **Test set.** All 512 test inputs, used for every task. So $\varepsilon$ is the exact population loss on the unseen half of the input space, not a sample.
- **Output and loss.** The 10 output bits are predicted jointly. Loss per example is $\sum_{j=0}^{9} -\log_2 p_\theta(y_j \mid x)$ bits. Outputs are deterministic, so the Bayes floor is $L^\star = 0$, and $\varepsilon$ is the held-out loss in **bits per example** (chance is 10 bits).
- **Finite training inputs.** Each task has 512 distinct training inputs, so $T$ counts examples seen with repetition: passes over a finite set, not fresh draws. Every report gives $T$ in examples and in target passes, $\alpha T / 512$.

### 2.2 Exact complexity and its correctness checks

**Breadth-first search (BFS).** Level 0 is the identity. Level $L$ applies every primitive to every level $L-1$ function. Candidates are generated in the order (canonical program of the parent, then primitive index). Any candidate whose truth table equals an already-seen function is dropped. Equality is decided exactly: a 64-bit hash for sorting, then full-row comparison inside equal-hash groups, and a collision with unequal rows raises an error. The first occurrence is kept, which makes its program the lexicographically smallest shortest program (its **canonical program**). The search also accumulates $n_{\text{short}}(f)$, the number of distinct shortest programs. Measured on 10-18-2026 (numpy, one core): **12, 112, 988, 8,543, 73,146 and 622,876** new functions at lengths 1–6 (705,678 including the identity), in 49 s. An unchunked prototype peaked at 5.2 GB of RAM; candidate generation in chunks of 8,192 parents keeps the peak under 4 GB. Level 7 (about 5.3 million functions, estimated) is feasible but not needed.

**Correctness checks (unit tests).**

- Each primitive matches a plain per-bit Python loop on all 1,024 inputs.
- `GRAY`∘`PREFIX_XOR` and `PREFIX_XOR`∘`GRAY` are the identity. `NEG` equals `INC`∘`NOT`, so `NEG` has $K = 1$, and the sequence (`NOT`, `INC`) is deduplicated into it. `REV`, `NOT`, `SWAP_HALVES` and `SWAP_ADJ` are involutions, and `ROTL1`∘`ROTR1` is the identity.
- The level counts above are reproduced exactly.
- **Brute force:** enumerating all $\sum_{L \le 5} 12^L = 271{,}453$ programs (including the empty one) gives the same set of functions, the same $K$, the same canonical programs and the same $n_{\text{short}}$ for every function with $K \le 5$. A prototype on 10-18-2026 confirmed this for $K \le 4$ (9,656 functions).
- The canonical program of every selected task reproduces its stored truth table.

### 2.3 Eligibility filters and covariates

**Filters (all exact, over the 1,024 inputs):**

1. **Output balance:** the mean over inputs and the 10 output bits of $f_j(x)$ is in $[0.3, 0.7]$.
2. **Dependence:** $f$ depends on at least 8 of the 10 input bits. Input bit $i$ counts if $f(x) \ne f(x \oplus 2^i)$ for some $x$.
3. **Train-identifiability:** the restriction of $f$ to the 512 training inputs differs from the restriction of every other function with $K' \le K$. So the training data cannot tell $f$ apart from a shorter program, and an Occam learner that picks the shortest consistent program would recover $f$ exactly. This tightens the brief's rule (shorter programs only) to include equal-length rivals. On a prototype split (10-18-2026), eligible functions clashing with a shorter function numbered 0, 0, 4 of 8,492, 33 of 19,795 sampled and 54 of 19,750 sampled at $K = 2..6$; equal-length clashes numbered 0, 0, 6, 61 and 81. The filter rarely binds, but it is enforced and its counts are reported.
4. **Distinctness:** the truth table differs from, and is not the bitwise complement ($f \oplus 1023$) of, every task already chosen.

**Covariates (exact).**

- *Sensitivity* $s(f)$: the mean over the 10 input bits $i$ and the 10 output bits $j$ of $\Pr_x[f_j(x \oplus 2^i) \ne f_j(x)]$, in $[0, 1]$. It equals $\tfrac{1}{10}$ times the mean Fourier degree of the output bits: $s(f) = \tfrac{1}{100}\sum_j \sum_S \lvert S\rvert\,\hat g_j(S)^2$ with $g_j = (-1)^{f_j}$ ([O'Donnell, *Analysis of Boolean Functions*, arXiv 2105.10386](https://arxiv.org/abs/2105.10386), the influence formula). A unit test checks the identity.
- *Affine indicator:* whether $f(x) \oplus f(0)$ is linear over GF(2). This is true for every program without `INC` or `NEG`.
- *Local rules (the quanta covariate):* for input $x$, the local rule $J_f(x)$ is the $10 \times 10$ binary matrix whose $(i, j)$ entry is $1$ iff flipping input bit $i$ flips output bit $j$. Rule frequencies $p_r = \lvert\{x : J_f(x) = r\}\rvert / 1024$ give the **local-rule entropy** $H_{\text{rule}} = -\sum_r p_r \log_2 p_r$, the number of distinct rules, and the rare-rule mass (the share of inputs whose rule occurs on fewer than 16 of the 1,024 inputs). Affine functions have a single rule, so $H_{\text{rule}} = 0$. `INC` has one rule per carry-chain length, with geometric frequencies. This is the DSL analog of Experiment 11's state-visitation tail. Each rule is a sub-skill used with frequency $p_r$. Note that $s(f)$ is the *mean* size of $J_f(x)$, while $H_{\text{rule}}$ is the *diversity* of $J_f(x)$.
- $n_{\text{short}}$; per-output-bit balance (10 values); the number of inputs depended on; and the maximum output agreement with any other selected task, $\max_g \Pr_{x,j}[f_j(x) = g_j(x)]$.
- `visit_tail_exponent` is not defined for this family and is stored as null.

### 2.4 Sensitivity- and type-stratified task selection

A prototype on 10-18-2026 (all functions at $K \le 4$, 20,000 sampled at $K = 5, 6$) found that both confounds rise with $K$ among eligible functions:

| $K$ | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| median sensitivity | 0.14 | 0.17 | 0.19 | 0.21 | 0.23 |
| affine share | 0.66 | 0.51 | 0.38 | 0.31 | 0.21 |
| $H_{\text{rule}}$ of non-affine, 90th percentile (bits) | 3.96 | 4.45 | 5.39 | 6.14 | 7.00 |

So we select by design. Four sensitivity strata: S1 $= [0.08, 0.11)$, S2 $= [0.11, 0.14)$, S3 $= [0.14, 0.22)$, S4 $= [0.30, 1]$. Eligible counts at $K = 2$ are 42, 30, 20 and 20, and at $K = 3$ they are 186, 264, 274 and 250 (thousands at higher $K$). Each target/distractor **cell** also fixes the type: **S1 affine, S2 non-affine, S3 affine, S4 non-affine**. So every level has exactly 2 affine and 2 non-affine targets. This adds a type constraint to the brief's one-per-stratum rule, because with type left free the expected number of affine targets falls from about 2.6 at $K = 2$ to about 0.9 at $K = 6$. Eligible counts in the scarcest cells at $K = 2$ are S3-affine 16 and S4-non-affine 4 (2 are needed). S2 is entirely non-affine at every level.

**Procedure.** For each $K$, shuffle that level's functions with an rng seeded from (20261012, $K$), and walk the shuffled order in chunks of 4,096. A function that passes all filters and falls in a stratum is assigned as follows. If its cell still lacks a target or distractor and its type matches, the first such function becomes the **target** and the second the **distractor**. Otherwise, the first 4 of each stratum (any type) become **probe_pretrain** tasks and the next 4 become **fresh** tasks, for Experiment 14. Stop when every slot is filled. Taking the first eligible functions in a uniformly random order is uniform sampling without replacement within each cell. A prototype of this procedure gave median sensitivity 0.15, 0.16, 0.15, 0.14 and 0.15 at $K = 2..6$ over all 200 tasks (Spearman with $K$: −0.08).

| | $K=2$ | $K=3$ | $K=4$ | $K=5$ | $K=6$ | total |
|---|---|---|---|---|---|---|
| $K_{\text{bits}}$ | 11 | 14 | 20 | 23 | 27 | |
| targets (1 per cell) | 4 | 4 | 4 | 4 | 4 | 20 |
| distractors (1 per cell) | 4 | 4 | 4 | 4 | 4 | 20 |
| probe_pretrain (4 per stratum, not trained here) | 16 | 16 | 16 | 16 | 16 | 80 |
| fresh (4 per stratum, not trained here) | 16 | 16 | 16 | 16 | 16 | 80 |
| task tokens | | | | | | 40 |

Task ids are `dsl_k2_t1..t4`, `dsl_k2_d1..d4`, `dsl_k2_p01..p16` and `dsl_k2_f01..f16` (similarly for `k3`–`k6`), numbered in stratum order. Task tokens 0–19 go to targets (level order, then stratum order) and 20–39 to distractors. Probe and fresh tasks get no token here; Experiment 14 assigns its own.

### 2.5 The alignment construction

- **Mix.** For target $j$ and share $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$, a fraction $\alpha$ of examples come from target $j$. The remaining $1 - \alpha$ is split equally over the **same 20 distractors in every condition**, so the non-target part of the mix does not depend on the target's $K$. Targets are never distractors for each other.
- **Exact schedule.** Training runs to $T_{\max} = 2^{16}$ examples. In each octave interval ($[1, 2^{10}]$, then $(2^{k-1}, 2^k]$ for $k = 11..16$), exactly $\mathrm{round}(\alpha \times \text{interval length})$ slots go to the target, placed uniformly at random. The other slots are dealt to the 20 distractors in equal counts ($\pm 1$). So the target count at every checkpoint is within $\pm 4$ of $\alpha T$. The laws use the nominal $\alpha T$, and `mixes.json` records the exact counts.
- **Example order within a task.** Each task reads its 512 training inputs in successive passes, each pass a fresh permutation drawn from the data-order rng. So after $n$ examples of a task, every training input has been seen $\lfloor n/512 \rfloor$ or $\lceil n/512 \rceil$ times, and repetition carries no sampling noise. At $T = 2^{16}$ the target has made 128 passes at $\alpha = 1$ and 6.4 at $\alpha = 0.05$. Each distractor has made 6.1 passes at $\alpha = 0.05$ and none at $\alpha = 1$.
- **What $T$ counts.** $T$ counts all examples seen, target and distractor. Compute per run is identical across $\alpha$. Batches hold 64 examples. Tasks are mixed within a batch, and every example has 11 tokens, so no padding or length schedule is needed.

### 2.6 Model, optimizer and checkpoints

- **Transformer (primary).** These are Experiment 11's blocks: 2 layers, $d_{\text{model}} = 64$, 4 heads, multilayer perceptron (MLP) width 256, pre-LayerNorm, GELU, and learned position embeddings (11 positions). Attention is computed with `torch.nn.functional.scaled_dot_product_attention` from one fused query-key-value linear layer. Unlike Experiment 11, attention is **bidirectional** (no causal mask). A word-to-word function has no reading order, and with a causal mask, output bit $j$ would see only input bits $0..j$ at its own position. The input is `[task token, x_0, …, x_9]` over a 42-token vocabulary (ids 0 and 1 for the bit values, $2 + t$ for task token $t$). Output bit $j$ is read at the position of $x_j$ by one shared 2-way head (64 → 2, Experiment 11's head). It has **103,618 parameters**: embeddings 2,688, positions 704, two blocks of 49,984, final LayerNorm 128, head 130. This count was checked by instantiating the model on 10-18-2026 (torch 2.14.1+cpu); it is 103,682 with one extra token, the null token Experiment 14 adds. `run.py --check-only` asserts it.
- **Optimizer.** As Experiment 11: AdamW, learning rate $10^{-3}$, weight decay 0.01, betas (0.9, 0.999), batch 64, gradient-norm clip 1.0. There is a linear warmup over 8 steps, after which the rate stays constant. The rate is not tuned per condition.
- **Checkpoints.** $T \in \{2^{10}, 2^{11}, \dots, 2^{16}\}$, which is steps 16 to 1,024: 7 checkpoints along one constant-rate trajectory. Experiment 11's first checkpoint, $2^9$ (step 8, the end of warmup), is dropped for two reasons. With 10 output bits every target is still at chance there, a pre-power-law plateau. And 8 weight checkpoints per run would put artifacts over 2 GB (Section 7). The pilot (Section 2.8) may move the window under fixed rules.
- **Train-fit diagnostic.** At every checkpoint the loss on the 512 training inputs (`train_bits`) is also recorded. A run that fits training data but not test data is *memorizing without generalizing*. A run that does not fit training data is *not fitting* (optimization or expressivity). The two failure modes are counted separately.
- **Secondary arm R768 (data repetition).** Train on $\pi[0{:}768]$ (768 inputs) and test on $\pi[768{:}1024]$ (256 inputs). Distractors use the same 768 inputs. This runs all 20 targets at $\alpha \in \{1.0, 0.15\}$, seeds 12101–12404, with the same schedule and initialization. That is 160 runs, started only if at least 60 minutes remain under the wall cap after the main grid. At equal $T$ each input is repeated 1.5 times less often. It is compared with the main runs on the same 256 test inputs. The train-identifiability filter holds automatically, because 768 inputs contain the 512.
- **Exploratory arm MLP (no gate).** An MLP on [one-hot task token (40), 10 input bits]: hidden layers 256 and 256 with GELU, then 20 logits read as 10 two-way outputs. It has 83,988 parameters (instantiated 10-18-2026). It runs at $\alpha = 1$ for all 20 targets with seeds 12101–12404 (80 runs) and the same optimizer and checkpoints, only if at least 10 minutes remain. It is a learner with no attention routing, so it shows whether a $K$ effect is specific to the transformer.

### 2.7 Seeds, grid and splits

- **Training seeds** 12101, 12202, 12303, 12404, 12505, 12606, 12707, 12808. The seed sets the initialization (`torch.manual_seed(seed)` before building the model, so all cells of a seed share it) and the data order. Tasks and splits are fixed by task seed 20261012 and shared by all seeds. Analysis seed 29101226; 20,000 bootstrap draws.
- **Runs.** 5 levels × 4 targets × 4 $\alpha$ × 8 seeds = **640 runs**, each 1,024 steps. That makes 4,480 checkpoint rows and 640 × 7 × 512 = 2.29 million held-out predictions, each of 10 bits.
- **Fit and held-out cells.**

| Axis | Fit | Held-out |
|---|---|---|
| $K$ (program length) | 2, 3, 5, 6 | 4 (interpolation, the middle level) |
| $\alpha$ | 0.05, 0.4, 1.0 | 0.15 (interpolation) |
| $T$ | $2^{10} … 2^{15}$ | $2^{16}$ (extrapolation) |

A point is (seed, target, $\alpha$, $T$). Per seed there are 560 points: 288 fit points (16 targets × 3 $\alpha$ × 6 $T$) and **272 held-out points**, the union of the $K = 4$, $\alpha = 0.15$ and $T = 2^{16}$ slices. Over 8 seeds that is 2,176 held-out points.

### 2.8 Calibration pilot (disclosed; seed 12909, never reused)

Run one seed at $\alpha = 1$ for all 20 targets to $T = 2^{17}$: 20 runs, about 12 minutes. Before the freeze, apply these rules mechanically and record them in `CKPT`:

- **P1.** If the median over targets of $T^\star(1.0\ \text{bits})$ is above $2^{14}$, the primary $\varepsilon^\star$ becomes 3.0 bits, and 1.0 and 0.3 bits become the secondary levels.
- **P2.** If more than 5 of the 20 targets are already at or below 1.0 bits at $T = 2^{11}$, shift the whole window down one octave, to $2^9 … 2^{15}$.
- **P3.** Report the measured step time and checkpoint size. If the projected main grid exceeds 90% of the wall cap, or projected artifacts exceed 2 GB, stop and ask Brando. Do not cut seeds silently.
- **P4.** If more than 10 of the 20 targets are still above 3.0 bits at $T = 2^{17}$, the design is underpowered for this learner. Stop, record it, and ask Brando before freezing. Do not change the model or the learning rate without his sign-off.

The pilot also reports how many targets improve between $2^{16}$ and $2^{17}$, and the train-fit census. The value of G8a (Section 5), a property of the task set that `--check-only` computes, is disclosed in the same paragraph. Pilot outcomes are never pooled with the gated runs.

## 3. Measurements

### 3.1 Learning curves

Per checkpoint, written to `metrics.csv` with one row per (arm, seed, target, $\alpha$, $T$):

- `excess_bits` $= \varepsilon$: the mean over the 512 test inputs of $\sum_j -\log_2 p_\theta(y_j)$, computed in float32 and accumulated in float64.
- `accuracy` (all 10 bits right under greedy decoding); `bit_accuracy`; `n_test` = 512 (256 in R768).
- `excess_bits_by_output_bit` (10 values); `train_bits` and `train_accuracy` on the 512 training inputs.
- `target_examples_seen`, `distractor_examples_seen`, `target_passes`; `step`, `T`, wall time.
- Per-input bits for every checkpoint, saved to `predictions/<seed>_<target>_<alpha>.npz` (test: 7 × 512, train: 7 × 512, float32).

**Derived per run:** $T^\star(\varepsilon^\star)$ for $\varepsilon^\star \in \{3.0, 1.0, 0.3\}$ bits per example (the same per-output-bit levels as Experiment 11's 0.3, 0.1 and 0.03). Take the first checkpoint with $\varepsilon \le \varepsilon^\star$ and interpolate $\log_2 \varepsilon$ linearly in $\log_2 T$ from the checkpoint before it.

- *Right-censoring.* If the run never reaches $\varepsilon^\star$, $T^\star$ is recorded as $T_{\max}$ with a flag. *Left-censoring.* If the first checkpoint is already below $\varepsilon^\star$, $T^\star$ is recorded as $T_{\min}$ with a flag.
- *Plateau diagnostic.* Flag a run if $\varepsilon(2^{16}) \ge 0.8\,\varepsilon(2^{14})$ while $\varepsilon(2^{16}) \gt 1.0$ bits. Split flagged runs into *memorizing* (`train_bits` $\le 0.1$) and *not fitting* (`train_bits` $\gt 1.0$).

**Resolution floor.** $\varepsilon_{\text{res}} = 10^{-2}$ bits per example ($10^{-3}$ per output bit, as in Experiment 11). Points below it are never dropped. They enter fits through a censored residual (Section 4.1).

### 3.2 Complexity proxies (per task, all 200 tasks)

- **BDM**, primary: pybdm's `BDM(ndim=1)` (pybdm 0.1.0, installed and checked on 10-18-2026 with numpy 2.5.4). It uses the CTM table for binary strings of length 12 with the default `PartitionIgnore`. Its input is the 10,240-bit **row-major** truth table: inputs $x = 0..1023$ in order, each output written bits 0..9. The last 4 bits are dropped by the partition. One evaluation takes about 15 ms.
- **BDM, bit-plane order** (secondary): all 1,024 values of output bit 0, then bit 1, and so on.
- **Compressors:** `zlib` level 9 and `lzma` preset 9 byte lengths of the row-major table packed with `np.packbits` (1,280 bytes), and `zlib` length of the uint16 little-endian output array (2,048 bytes).
- **Prototype (10-18-2026, same selection rules, prototype split, 200 tasks).** Spearman with $K$: BDM row-major −0.31, BDM bit-plane +0.21, zlib −0.40, lzma −0.28. Longer programs more often include `SHL1` or `SHR1`, which zero output bits and make the table compressible. The truth table of a bijection looks incompressible at the 12-bit-block and byte level, even when its program has length 2. So we **predict G8a and G8b `NOT_ESTABLISHED`**. Exact $K$ in a fixed language and block-level string complexity measure different things.
- **Fallback.** If pybdm fails to install or import, record the error in `CKPT`, compute the 12-bit block Shannon entropy as a labelled substitute, and report G8 as `NOT_ESTABLISHED` (not evaluable). Never substitute silently.

**Per task** (in `tasks.json`): $K$, $K_{\text{bits}}$, the canonical program, the full truth table, $n_{\text{short}}$, balance (overall and per bit), inputs depended on, sensitivity and stratum, affine indicator, $H_{\text{rule}}$, rule count, rare-rule mass, the proxies, and the maximum agreement with other tasks.

## 4. Laws and analysis plan

### 4.1 Fitting

All six laws from the overview are fit with $T$ = total examples and nominal $\alpha$:

```math
\begin{aligned}
\texttt{M0\_scale}&:\ \varepsilon = A\,T^{-\gamma} \\
\texttt{M\_align\_eff}&:\ \varepsilon = A\,(\alpha^{\eta}T)^{-\gamma} \\
\texttt{M\_align\_add}&:\ \varepsilon = a + c\,T^{-\gamma} - d\,\alpha \\
\texttt{M\_K\_coef}&:\ \varepsilon = A\,K^{\kappa}(\alpha^{\eta}T)^{-\gamma} \\
\texttt{M\_K\_exp}&:\ \varepsilon = A\,(\alpha^{\eta}T)^{-(\gamma_0+\gamma_1\log_2 K)} \\
\texttt{M\_K\_both}&:\ \varepsilon = A\,K^{\kappa}(\alpha^{\eta}T)^{-(\gamma_0+\gamma_1\log_2 K)}
\end{aligned}
```

- **Primary $K$ regressor:** $K_{\text{bits}} \in \{11, 14, 20, 23, 27\}$, the unit of the MDL prediction. Every fit is repeated with $K$ = program length and reported alongside. The range is narrow ($\log_2(27/11) = 1.30$, against 4.14 in Experiment 11), so the fitter works internally with $\log_2(K/18.04)$ for conditioning, where 18.04 is the geometric centre of $K_{\text{bits}}$. It reports $\gamma_0$ in the overview's uncentered form. $\kappa$ and $\gamma_1$ do not depend on centering.
- **Per-seed fits.** Each law is fit separately per seed on that seed's 288 fit points by nonlinear least squares on $\log_2 \varepsilon$ (SciPy `least_squares`, trust-region reflective). The censored residual is $r = \log_2 \hat\varepsilon - \log_2 \varepsilon$ when $\varepsilon \ge \varepsilon_{\text{res}}$, and $r = \max(0, \log_2 \hat\varepsilon - \log_2 \varepsilon_{\text{res}})$ otherwise.
- **Bounds.** Every exponent $\gamma$, including $\gamma_0 + \gamma_1\log_2 K$ at $K_{\text{bits}} \in [11, 27]$, lies in $[10^{-3}, 3]$; $\eta \in [0, 3]$; $\kappa \in [-5, 5]$.
- **Starts.** 64 starts form a 4 × 4 × 4 grid: $\gamma$ or $\gamma_0 \in \{0.1, 0.3, 0.6, 1.2\}$; $\eta \in \{0.25, 0.75, 1.25, 2.0\}$; and a third axis that is $\kappa \in \{-0.5, 0.5, 1, 2\}$ for `M_K_coef`, $\gamma_1 \in \{-0.3, -0.1, 0, 0.1\}$ for `M_K_exp`, and $(\kappa, \gamma_1) \in \{(0,0), (1,0), (0,-0.2), (1,-0.2)\}$ for `M_K_both`. The $\gamma_1$ grid is wider than Experiment 11's because the $\log_2 K$ range is narrower. Laws with fewer exponents use the sub-grid. $\log A$ starts at its least-squares value given the exponents.
- **`M_align_add`** is fit by least squares on $\varepsilon$ in linear space, as in the alignment-scaling paper with Varun, with $a, c, d$ free and $\gamma \in [10^{-3}, 3]$. It is scored on the same $\log_2$ scale. A forecast $\le 0$ is invalid and is clipped to $\varepsilon_{\text{res}}$ for RMSE.
- **Independent fitter.** `independent_fit.py` profiles $\log A$ out analytically and minimizes over the exponents with L-BFGS-B from the same starts. Its objective and held-out forecasts must match `laws.py` to $10^{-4}$ relative before anything is reported.

### 4.2 Held-out scoring

Forecasts for all 272 held-out points per seed are written to `forecasts_frozen.json`, and its SHA-256 is logged in `CKPT`, before any held-out row is read. `analyze.py` reads fit and held-out rows in separate stages. The score is the root-mean-square error (RMSE) of the censored residual on $\log_2 \varepsilon$, pooled over seeds. Relative reduction of law B over law A is $1 - \text{RMSE}_B/\text{RMSE}_A$. An invalid forecast is nonfinite or $\le 0$.

### 4.3 Uncertainty

All intervals are 95% percentile intervals from a seed-block bootstrap: 20,000 draws that resample the 8 seeds with replacement (analysis seed 29101226). Points within a seed are never treated as independent. A parameter's estimate is the mean over seeds of per-seed estimates. Where a gate says so, a **two-stage bootstrap** also resamples the 4 targets within each $K$ level in every draw, which asks whether a result generalizes beyond these 20 functions. Seed-only intervals are conditional on the sampled targets, and the report says so. Gates on the 200-task proxies (G8a) resample tasks within each level.

### 4.4 Data multipliers and elasticities

- **Complexity multiplier (model-free).** At $\alpha = 1$ and $\varepsilon^\star$: $M_K = \mathrm{GM}_j\, T^\star_j(K{=}6) \,/\, \mathrm{GM}_j\, T^\star_j(K{=}2)$, where GM is the geometric mean over the 4 targets of a level. Each level has one target per (stratum, type) cell, so $M_K$ is also the geometric mean of 4 cell-paired ratios, which match sensitivity stratum and type by design. It is computed per seed, and the point estimate is the geometric mean over seeds. Censoring at either end makes $M_K$ smaller, which is the conservative direction for C1.
- **Elasticities (law-based).** Fit `M_K_both` per seed on **all** 560 points (descriptive). Then

  ```math
  \log T^\star(K,\alpha) = \eta\log\tfrac{1}{\alpha} + \frac{\log A + \kappa \log K - \log \varepsilon^\star}{\gamma(K)}, \qquad \gamma(K) = \gamma_0 + \gamma_1\log_2 K .
  ```

  $e_\alpha = \eta$, and $e_K = \partial \log T^\star / \partial \log K$ is evaluated analytically at $K_{\text{bits}} = 18.04$. $E = \bar e_\alpha / \bar e_K$ is the ratio of seed means, with $E = +\infty$ in a draw where $\bar e_K \le 0$.
- **Units.** In program length, $E_{\text{len}} = E_{\text{bits}} / 0.842$, where 0.842 is the least-squares slope of $\log K_{\text{bits}}$ on $\log K$ over the five levels. The gate requires both, so here the bits unit is the binding one, the opposite of Experiment 11.
- **Model-free check.** $E_{\text{mf}}$: per seed, regress $\log_2 T^\star_{j,\alpha}$ on $\log_2(1/\alpha)$ and $\log_2 K_{\text{bits},j}$ over the uncensored (target, $\alpha$) pairs (of 80). The censored count is reported.
- **Accuracy dependence.** $E$ is reported at $\varepsilon^\star \in \{3.0, 1.0, 0.3\}$ bits and as a curve over $[10^{-2}, 10]$ bits, with the flip point $\varepsilon_{\text{flip}}$ where $E = 1$ if it lies in that range.
- **Range ratio** (the overview's wording of C2, reported, no gate): $R = [T^\star(\alpha{=}0.05)/T^\star(\alpha{=}1)] / M_K$. Because $\ln 20 = 3.00 \gt \ln(27/11) = 0.90$, $R \gt 1$ is a much weaker requirement than $E \gt 1$.
- **Passes.** Every $T^\star$ table is also given in target passes, $\alpha T^\star / 512$.

### 4.5 Alignment comparison (reported, no gate)

`M_align_eff` versus `M_align_add` on the joint held-out set: RMSE, relative reduction with its interval, and invalid counts. Also $\eta$ with its interval from `M_align_eff` and from the winning $K$ law; $\eta \lt 1$ means distractors transfer, and $\eta \gt 1$ means they interfere. The per-target $\eta_j$ (from per-target `M_align_eff` fits) is reported against $K$ and against primitive overlap with the distractor pool: the number of the target's canonical primitives that appear in any distractor's canonical program.

### 4.6 Mechanism and confound analyses

- **Per-target exponent.** For each target, fit $\varepsilon = A_j T^{-\gamma_j}$ to its $\alpha = 1$ curve, per seed, with the censored residual; $\gamma_j$ is the mean over seeds.
- **Quanta test (G7).** Spearman correlation of $\gamma_j$ with $H_{\text{rule},j}$ across the 20 targets, and across the 10 non-affine targets alone. The quanta picture predicts a negative sign: more diverse local rules, with rare ones, mean a slower power law. The partial Spearman controlling for $\log K_{\text{bits}}$ is reported.
- **$K$ versus sensitivity versus type (reported).** Regress per-target $\log_2 T^\star_j(\alpha{=}1, 1.0\ \text{bits})$, averaged over seeds, on standardized $\log K_{\text{bits}}$, standardized sensitivity and the affine indicator. Report all coefficients with seed-block and two-stage intervals, and variance inflation factors. Also report $M_K$ separately for affine and non-affine cells.
- **Simplicity bias versus program length (reported).** Spearman of per-target $T^\star$ with $K$, BDM, zlib and lzma, and the partial Spearman of $T^\star$ with $K$ given zlib, and with zlib given $K$.
- **Train-fit census.** Plateau, memorizing and not-fitting counts by level and type.
- **Repetition (R768).** For matched (target, $\alpha$, seed, $T$), the ratio $\varepsilon_{768}/\varepsilon_{512}$ on the shared 256 test inputs. Report its median by $\alpha$, and $M_K$ and $\eta$ (two $\alpha$ levels) in both arms. If the median ratio at $T = 2^{16}$, $\alpha = 1$ is below 0.5, the report must say that the main results partly measure generalization from a finite input set.
- **MLP arm.** $M_K$ and the train-fit census, next to the transformer's.
- **Per-output-bit loss** at $T = 2^{16}$ by level, to see whether failures concentrate on high bits (long carries).

## 5. Prospective pass/fail gates

Every gate ends `SUPPORTED` or `NOT_ESTABLISHED`, and all are reported with equal prominence. Primary gates are G1–G4; secondary gates are G5–G9. All thresholds are fixed at the freeze. The primary $\varepsilon^\star$ is 1.0 bits per example unless P1 fires.

1. **G1, complexity is real (C1, primary).** $M_K$ at $\alpha = 1$ and the primary $\varepsilon^\star$. `SUPPORTED` iff $M_K \ge 2.0$, its seed-block 95% lower bound is $\gt 1.0$, and its two-stage (seeds and targets) 95% lower bound is $\gt 1.0$.
2. **G2, complexity adds predictive value beyond alignment (primary).** Compare the $K$ law against `M_align_eff` on the joint held-out set. The $K$ law is whichever of `M_K_coef` and `M_K_exp` has the lower fit-cell RMSE, chosen before held-out rows are read. `SUPPORTED` iff the relative RMSE reduction is $\ge 10\%$, its 95% lower bound is $\gt 0\%$, the $K$ law's RMSE is also lower on the $K = 4$ slice alone, and there are 0 invalid forecasts.
3. **G3, which law (C3, primary; two mutually exclusive sub-gates).**
   - **G3a, coefficient (MDL).** `SUPPORTED` iff, in `M_K_both`, $\kappa$'s 95% interval lies above 0 and $\gamma_1$'s interval contains 0, and the held-out RMSE of `M_K_coef` is $\le$ that of `M_K_exp`.
   - **G3b, exponent (quanta).** `SUPPORTED` iff, in `M_K_both`, $\gamma_1$'s 95% interval lies below 0, and `M_K_exp` cuts held-out RMSE relative to `M_K_coef` by $\ge 5\%$ with a 95% lower bound $\gt 0\%$.
   - If both are `NOT_ESTABLISHED`, C3 is reported as **unresolved**.
4. **G4, alignment dominates in units of data (C2, primary).** `SUPPORTED` iff all three hold: the law-based $E$ at the primary $\varepsilon^\star$ is $\gt 1$ with a seed-block 95% lower bound $\gt 1$; the same holds in both $K$ units (bits and program length); and the model-free $E_{\text{mf}}$ point estimate is $\gt 1$. If more than 50% of (target, $\alpha$) pairs are censored at the primary $\varepsilon^\star$ in the median seed, $E_{\text{mf}}$ is evaluated at 3.0 bits instead, and this fallback is fixed now.
5. **G5, complexity gap shrinks with training (C2, secondary).** At $\alpha = 1$, $g(T) = \bar\varepsilon_{K=6}(T) - \bar\varepsilon_{K=2}(T)$ in bits, the difference of target means, and $\rho = g(2^{16})/g(2^{12})$, the ratio of seed means. `SUPPORTED` iff $g(2^{12}) \gt 0$, $\rho \le 0.5$, and the 95% upper bound of $\rho$ is $\lt 1.0$. The gap in $\log_2 \varepsilon$ is reported next to it.
6. **G6, alignment sets a floor (secondary).** `M_K_floor` adds $a_1(1-\alpha)$ with $a_1 \ge 0$ to the G2 $K$ law. `SUPPORTED` iff $a_1$'s 95% lower bound is $\gt 0$ and `M_K_floor` cuts joint held-out RMSE relative to the G2 $K$ law by $\ge 10\%$ with a 95% lower bound $\gt 0\%$. The prediction is `NOT_ESTABLISHED`, because the task token removes label conflict.
7. **G7, quanta mechanism (secondary).** `SUPPORTED` iff Spearman($\gamma_j$, $H_{\text{rule},j}$) over the 20 targets is $\le -0.5$ with a seed-block 95% upper bound $\lt 0$, **and** the point estimate over the 10 non-affine targets alone is $\lt 0$. The two-stage interval is reported next to it. Requiring the sign within the non-affine targets stops the affine/non-affine split (10 ties at $H_{\text{rule}} = 0$) from carrying the gate alone.
8. **G8, CTM/BDM cross-check (secondary; two sub-gates, never primary).**
   - **G8a, BDM tracks exact $K$** (a property of the task set; computed by `--check-only` and disclosed before the freeze). `SUPPORTED` iff Spearman(row-major BDM, $K$) over the 200 tasks is $\ge 0.5$ with a task-bootstrap 95% lower bound $\gt 0$.
   - **G8b, BDM tracks learning cost** (prospective). `SUPPORTED` iff Spearman(row-major BDM$_j$, $T^\star_j$) at $\alpha = 1$ and the primary $\varepsilon^\star$ over the 20 targets is $\ge 0.5$ with a seed-block 95% lower bound $\gt 0$.
   - The same statistics for bit-plane BDM, zlib and lzma are reported without verdicts. The prototype predicts both sub-gates `NOT_ESTABLISHED` (Section 3.2).
9. **G9, cross-language agreement with Experiment 11 (secondary).** This reads Experiment 11's `analysis.json` (its frozen gates; Experiment 11 runs first, and its outcome is known at this freeze, which is disclosed). `SUPPORTED` iff (i) Experiment 11's G3 verdict is decided (G3a or G3b) and this experiment's G3 verdict is the same; and (ii) the `M_K_both` parameter carrying that picture ($\kappa$ for G3a, $\gamma_1$ for G3b, both in $K_{\text{bits}}$ units) has the same sign in both experiments, with a ratio of point estimates (this one over Experiment 11's) in $[0.5, 2.0]$. If Experiment 11's G3 is unresolved or its results are missing, G9 is `NOT_ESTABLISHED` with the reason stated.

## 6. What each outcome would mean

| Outcome | Interpretation |
|---|---|
| G1, G2, G4 and G3a `SUPPORTED` | Brando's hypothesis in its coefficient form, in a second language. Complexity costs a fixed multiple of data, and alignment's exchange rate is larger. |
| G1, G2 and G3b `SUPPORTED` | Complexity is real and grows with accuracy. C2 holds only above $\varepsilon_{\text{flip}}$, and the report must say where it flips. |
| G1 `NOT_ESTABLISHED` | Shortest-program length did not measurably raise the data bill for this learner and budget. Check the type/sensitivity regression, the train-fit census and R768 before concluding anything. |
| G1 `SUPPORTED`, but sensitivity or the affine indicator absorbs $\log K$ in Section 4.6 | The cost tracks parity degree or carries, not program length. C1 should be restated in those terms. |
| Many runs memorize (train fit, test not), mostly at high $K$ | Half the truth table does not pin the function down for this learner, even though it is identifiable. That is a generalization cost, not an optimization cost. |
| Many runs fail to fit training data at high $K$, while the MLP fits them | An architecture limit of the 2-layer transformer, not a data cost; $\gamma_1 \lt 0$ from such runs must not be read as quanta. |
| R768 median ratio well below 1 | Repetition, not only complexity or alignment, shapes the curves. $\eta$ and $\gamma$ are then partly properties of a 512-input world. |
| G4 `NOT_ESTABLISHED` with $E \lt 1$ | Per log unit, complexity costs more data than misalignment, which contradicts C2 for this learner. |
| $\eta \lt 1$ ($\eta \gt 1$) with its interval below (above) 1 | Distractors built from the same primitives transfer (interfere). |
| G7 `SUPPORTED` | Local-rule diversity predicts exponents: direct evidence for the quanta mechanism in programs. |
| G8a/G8b `NOT_ESTABLISHED`, as predicted | Block-level string complexity (CTM/BDM) and compressors do not see program structure that exact $K$ in a fixed language sees. Do not use them as stand-ins for $K$ on structured tasks. |
| G8b `NOT_ESTABLISHED` while compressibility predicts $T^\star$ better than $K$ | Simplicity bias toward compressible truth tables, not program length, governs learning cost here. |
| G9 `SUPPORTED` | Automata and programs agree on how complexity enters the law, so C3 is not an artifact of one language. |

## 7. Compute estimate and hardware

The target is a CPU-only Linux box with 4 cores, 15 GB RAM and no GPU, running 4 worker processes with 1 torch thread each. Measured on 10-18-2026 on a comparable box (Intel Xeon 2.20 GHz, 2 physical cores with 4 hardware threads, 15 GB, torch 2.14.1+cpu, fused attention), with 4 concurrent processes:

- **Per step:** about 65 ms for one training step of 64 examples of 11 tokens with 10 outputs each (30 ms when running alone). Using `torch.nn.MultiheadAttention` instead raised this to about 77 ms, so the fused call is specified.
- **Per checkpoint evaluation:** about 0.4 s (512 test plus 512 train inputs).
- **Per run:** 1,024 steps plus 7 evaluations plus saving, about 70 s.
- **Main grid:** 640 runs / 4 workers × 70 s ≈ **3.1 h** (about 2 h on 4 physical cores).
- **Other costs:** BFS, selection, covariates and proxies take about 3 min; the pilot about 12 min; R768 (160 runs) about 47 min; the MLP arm (80 runs, about 10 ms per step) about 4 min; `verify.py` about 15 min.
- **Wall cap:** `max_wall_seconds = 18000` (5 h) for the full run, including the secondary arms, which start only if their time margins remain. At the cap the run stops cleanly. Finished cells are atomic and are never retrained. A `--resume` into the same output directory is allowed only once, and it is logged in `CKPT`.
- **Memory:** under 4 GB in the parent process during BFS, before workers start; under 0.5 GB per worker. Workers receive only the 40 trained truth tables, never the BFS table.
- **Disk:** a checkpoint is 103,618 × 4 bytes = 414 KB. The main grid stores 4,480 checkpoints: **1.86 GB**. Predictions take about 20 MB and data under 1 MB. The secondary arms store predictions but **no weight checkpoints**, which keeps the total at about 1.9 GB, under 2 GB. Their rows are verified by deterministic retraining of a sample (Section 8). No tar copy is made, since one would double disk use.

## 8. Artifacts and verification

Under `experiments/12_dsl_shortest_program_complexity_vs_alignment/expt_v1/results/`, following the shared output contract that Experiments 14 and 15 read:

- `tasks.json`: all 200 tasks with `task_id`; `family: "dsl"`; `role` (`target`, `distractor`, `probe_pretrain`, `fresh`); `task_token` (null for probe and fresh tasks); `K`; `K_bits`; `definition` (`primitives`: the canonical program in application order, `word_bits: 10`, `order: "x -> p_K(...p_1(x))"`); `truth_table` (1,024 ints); `output_balance`; `output_bit_balance`; `n_depends`; `sensitivity`; `sensitivity_stratum`; `is_affine`; `local_rule_entropy`; `n_local_rules`; `rare_rule_mass`; `n_shortest_programs`; `bdm_rowmajor`; `bdm_bitplane`; `zlib_bytes`; `lzma_bytes`; `zlib_u16_bytes`; `max_agreement`; and `visit_tail_exponent: null`.
- `data/<task_id>_{train,test}.npz` for all 200 tasks: `x` (uint16 inputs in split order: 512 train, 512 test), `y` (uint16 outputs), `bits_x` and `bits_y` (uint8, shape (n, 10), bit $i$ in column $i$). Every training input of every task is used by every mix, since even at $\alpha = 0.05$ the target makes 6.4 passes. A consumer that needs more than 512 examples per task, or examples no mix used, must sample with replacement from the train partition with `data.example_stream`, and content repeats. Experiment 14 uses 2,048 examples per task and Experiment 15 uses 4,096 scored units plus 2,048 references.
- `mixes.json`: for each (target, $\alpha$, seed), the per-task counts and target passes at every checkpoint, and the data-order seed.
- `metrics.csv` (4,480 main rows plus the secondary-arm rows, labelled by `arm`), `predictions/*.npz`, `tstar.csv`, `fits.json`, `forecasts_frozen.json`, `analysis.json`, `bfs_counts.json` (level counts and timing).
- `checkpoints/<seed>_<target_id>_<alpha>/T_<T>.npz`: float32 weights as plain numpy arrays, with no pickle, main grid only. They stay on the run box, out of git (experiment-local `.gitignore`), and each file's SHA-256 is listed in `ARTIFACT_SHA256.json`. Training is bitwise deterministic for a given seed, torch version and one thread, which a unit test checks.
- `manifest.json`: source and config SHA-256 (Experiments 14 and 15 check the hashes of `dsl.py`, `data.py` and `model.py` against it), library versions including pybdm, host, wall time and the pilot record.

**`verify.py`** must report PASS before any result is written up. It:

- re-hashes the sources;
- reruns the BFS from scratch, checks the level counts, and recomputes every task's $K$, canonical program and $n_{\text{short}}$;
- regenerates every task from the task seed and recomputes every filter, covariate and proxy;
- regenerates the split, data files and schedules, and compares them byte for byte;
- reloads every main-grid checkpoint into a fresh model and recomputes every main `metrics.csv` loss (test and train) with `max_abs_diff < 1e-9` bits;
- retrains 8 secondary-arm cells chosen with the analysis seed and checks bitwise-equal predictions;
- recomputes $T^\star$;
- refits all laws with both fitters (agreement within $10^{-4}$ relative);
- recomputes every gate from scratch.

## 9. Scope and limits

- $K$ is exact only for $\mathcal L_{\text{DSL}}$: these 12 primitives on 10-bit words, with this length code. It is not universal Kolmogorov complexity. Another primitive set would change which functions are simple. That is why Experiment 11 uses a different language and G9 compares them.
- The $K$ range is narrow (2–6, or 11–27 bits) because level 7 multiplies the search by about 9. Complexity elasticities rest on a factor of 2.45 in $K_{\text{bits}}$.
- There is one architecture, one model size, one learning rate and one word width. Bidirectional attention with per-bit readout is a choice; the MLP arm is the only other learner.
- With 512 training inputs per task, $T$ beyond about $2^{9}/\alpha$ is repetition. R768 measures how much that matters but does not remove it. There is no fresh-data version of this task family.
- With 4 targets per level, seed-only intervals are conditional on these 20 functions. G1 adds a two-stage interval for that reason.
- $\alpha$ is set by construction, the distractors are other programs with their own task tokens, and inputs are shared. This is the cleanest version of alignment, not the measured alignment of real data, which Experiments 14 and 15 address.
- CTM/BDM values depend on the serialization of the truth table, and BDM on a 10,240-bit string is dominated by 12-bit-block statistics. G8 tests these tools on this family only.

**TLDR-end:** [tdc: Expt 12 design] Twenty target functions on 10-bit words with exact shortest-program lengths 2–6 in a 12-primitive language (exhaustive breadth-first search, brute-force-checked, stratified by sensitivity and affine type) are each trained at target shares 0.05–1.0 against a fixed 20-function distractor pool. With 8 seeds, the 640 CPU runs of a 103,618-parameter transformer test on held-out cells whether complexity raises the data bill (C1), whether alignment still dominates in units of data (C2), and whether complexity moves the scaling coefficient or the exponent (C3). The design adds a local-rule quanta test, a data-repetition arm, a CTM/BDM cross-check, and an agreement check with Experiment 11's automata.
