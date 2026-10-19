# Protocol: Addition crossed, carry-chain length versus alignment (`16_carry_alignment_crossed_arithmetic`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/16_carry_alignment_crossed_arithmetic/PROTOCOL.md>

Status: DRAFT design, 10-18-2026, from Brando Miranda's complexity-versus-alignment idea. The running agent may revise it, then must freeze it in its own commit before any gated run; after the freeze nothing in Sections 2–5 changes.

Program overview: [complexity versus alignment, Experiments 11–16](../ideas/complexity_vs_alignment/README.md). Notation follows it: $\varepsilon$ excess loss in bits, $T$ training examples seen, $\alpha$ target share, $N$ non-embedding parameters, and the law family `M0_scale`, `M_align_eff`, `M_align_add`, `M_K_coef`, `M_K_exp`, `M_K_both` with parameters $A, \eta, \kappa, \gamma_0, \gamma_1$.

The experiment has two arms under one protocol:

- **16a** (CPU, central processing unit; multilayer perceptron on 3-digit addition, run by the coding agent; secondary);
- **16b** (GPU, graphics processing unit; decoder transformers on 8-digit addition at three model sizes, launched by Laolu or Brando on SNAP (Stanford Network Analysis Project) GPU nodes or Sherlock; primary).

## 1. Scientific motivation

**What this experiment is, and is not.** Experiments 11 and 12 vary exact complexity $K_{\mathcal L}$, the length of the shortest description in a fixed language. Addition cannot do that: every addition problem is solved by the same short program, so under any fixed description language its $K$ does not change with the carry pattern. Carry-chain length $c$, the longest run of consecutive carries in a problem, is a **structural demand**: how many sequential carry steps a solver must resolve on that instance ([COMPLEXITY.md](../archive/01_fixed_width_addition_exposure/COMPLEXITY.md); [predictability bridge, "Three different variables"](../ideas/predictability_bridge/README.md#three-different-variables)). Experiment 16 never calls $c$ complexity. It asks a narrower question that connects the program back to Experiments 01–09: **does a structural-demand variable enter the scaling law the same way exact $K$ does in Experiments 11 and 12?** The law variable is $s = c + 1$, so that $\log_2 s$ is defined at $c = 0$. All law names keep the program's `M_K_*` names with $K$ replaced by $s$.

**Hypothesis (Brando, 10-18-2026).** Complexity is a real learning cost, but alignment dominates once the model has trained on aligned data. Experiments 03 and 06 already showed that exposure can reverse the difficulty ranking of carry types at fixed data, including with exact answer matching and carry-marginal matching (Experiment 06: 22.35 and 17.74 percentage-point reversals). They did not fit a law in $T$ and $\alpha$, so they could not say whether carry demand moves the coefficient or the exponent, or how large its data multiplier is next to alignment's. Here we test:

- C1: the chain-length data multiplier is $\gt 1$;
- C2: alignment's multiplier range exceeds chain length's, and the chain-length gap shrinks with $T$ at $\alpha = 1$;
- C3: coefficient versus exponent.

We then compare the C3 verdict with Experiments 11 and 12.

**Competing predictions.**

| Picture | Prediction |
|---|---|
| Coefficient | Longer chains need more examples before the carry rule is learned, but once learned it applies everywhere. Constant multiplier: $\kappa \gt 0$, $\gamma_1 = 0$. |
| Exponent (quanta, [Hutter 2021](https://arxiv.org/abs/2102.04074); [Michaud et al. 2023](https://arxiv.org/abs/2303.13506)) | Each extra chain step is a rarer sub-skill: chains of length $\ge k$ are geometrically rarer under the natural distribution (Section 2.5), so $\gamma$ falls with $s$ ($\gamma_1 \lt 0$). |
| Alignment as effective data | $\varepsilon = A(\alpha^\eta T)^{-\gamma}$ with $\eta \approx 1$, and chain length adds little once $\alpha$ is known. |
| Format matters (mechanism) | Chain length is a demand only when the output format forces the model to resolve the chain before writing a digit (Section 2.4). In reversed-output format the chain becomes a local lookup, so its multiplier should shrink. |

**Why two arms.** 16a reuses the exact Experiment 06/08 setup: the 3-digit dataset with answer matching, the multilayer perceptron (MLP), and the optimizer. It links the laws directly to the earlier reversals, but its training set is a fixed 3,600 pairs reused over many epochs, and $c \le 2$. 16b is the clean test: fresh examples (no epochs), 8-digit numbers so that $c$ ranges over 0–8, transformers in the format used to teach arithmetic to small transformers ([Lee, Sreenivasan, Lee, Lee and Papailiopoulos 2023, "Teaching Arithmetic to Small Transformers", arXiv 2307.03381](https://arxiv.org/abs/2307.03381)), and three model sizes for the full law $L = a + bN^{-\beta} + c(s)\,T^{-\gamma(s)} - d\alpha$ of the [alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws) against its effective-data counterpart.

## 2. Experimental design

### 2.1 Arm 16a: 3-digit addition with an MLP (CPU, secondary)

**Dataset.** Exactly the `dataset()` construction of Experiment 08 (`experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/run.py`), with split seed `20261016`. It uses:

- unordered pairs $a \le b \lt 1000$ with $a + b \lt 1000$;
- carry cells from the units carry $c_0$ and the tens carry $c_1$: cell 0 = no carry, cell 1 = units only, cell 2 = tens only, cell 3 = both;
- the 180 answers that have at least 28 pairs in every cell;
- per (answer, cell): 4 test, 2 calibration and 2 probe pairs, the rest a training pool of at least 20 pairs.

The test set therefore has 2,880 pairs, 720 per cell.

**Strata and chain length.** No-carry = cell 0 ($c = 0$). Isolated = cells 1 and 2 ($c = 1$), always used in equal halves. Cascade = cell 3 ($c = 2$). Each stratum is in turn the **target**, and the other two are the distractors.

**$\alpha$ construction.**

- Each training set has 3,600 pairs, exactly 20 per answer. Experiment 06 used 40 per answer, but the pools hold only $\ge 20$ per (answer, cell). The largest single-cell demand here is 18 per answer (cascade target at $\alpha = 0.9$), so 20 per answer always fits.
- The target stratum gets $3600\alpha$ pairs. Each distractor stratum gets $1800(1-\alpha)$. Within the isolated stratum the units and tens cells split equally. For example, at $\alpha = 0.33$ the target gets 1,188 pairs, each distractor 1,206, and an isolated distractor 603 + 603.
- Per-answer quotas are $\lfloor 20\cdot\text{share}\rfloor$ per cell. The remaining units go out by a deterministic max-flow assignment, so every answer has exactly 20 pairs and every cell total is exact. Every grid cell was checked to give integer totals.
- Within a seed, each (answer, cell) pool is permuted once, and a set takes the first $k$ pairs. So sets are nested across $\alpha$ (common random numbers).

**Confound disclosed.** Answer histograms are matched exactly. Column-carry marginals cannot all be matched when the strata themselves are the manipulated variable. The units-carry rate equals the tens-carry rate, and both equal $s_{\text{iso}}/2 + s_{\text{casc}}$, where $s$ is the stratum's share. That rate is 0.5 at every $\alpha$ when isolated is the target. It is $3(1-\alpha)/4$ (0.675, 0.5025, 0.30, 0.075) when no-carry is the target, and $(1+3\alpha)/4$ (0.325, 0.4975, 0.70, 0.925) when cascade is the target. It is logged per mix.

**Grid.** $\alpha \in \{0.1, 0.33, 0.6, 0.9\}$. MLP hidden widths $w \in \{32, 64, 96, 128\}$. Architecture: input $60 \to w \to w \to 40$ with ReLU, as in Experiment 06. Parameters: 4,328 / 10,664 / 19,048 / 29,480, all counted as $N$. Seeds `16101, 16202, …, 16808` (8). That gives 3 targets × 4 $\alpha$ × 4 widths × 8 seeds = **384 runs**.

**Training.**

- AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch 256 drawn with replacement from the 3,600 pairs.
- Linear warmup over 50 steps, then constant.
- Checkpoints at 100, 300, 1,000 and 3,000 updates ($T$ = 25,600 to 768,000 examples, 7 to 213 epochs). This is a repeated-epoch regime, disclosed as the main limit of 16a.
- Same initialization per (seed, width) across targets and $\alpha$.

**Fit and held-out cells.**

| Coordinate | Fit | Held out |
|---|---|---|
| target $c$ | 0, 2 | 1 |
| $\alpha$ | 0.1, 0.6, 0.9 | 0.33 |
| $T$ (updates) | 100, 300, 1,000 | 3,000 |
| width | 32, 64, 128 | 96 |

**Pilot.** Seed `16909`, width 64, all 3 targets, $\alpha \in \{0.1, 0.9\}$ (6 runs, about 1 min). It confirms timing and applies the $\varepsilon^\star$ rule of Section 4.4.

### 2.2 Arm 16b: 8-digit addition with transformers (GPU, primary)

**Problems.** Ordered pairs $(a, b)$ with $0 \le a, b \lt 10^8$, written as zero-padded 8-digit numbers, so input length never varies with the operands. Chain length $c \in \{0, \dots, 8\}$ is the longest run of consecutive columns that produce a carry, counting the carry out of the top column (as in Experiment 06's `carry_chain`).

**Format (primary: plain, most significant digit first).** Vocabulary of 13 tokens (digits 0–9, `+`, `=`, `$`). Each example is

```text
$ a7 a6 … a0 + b7 b6 … b0 = s8 s7 … s0 $
```

That is 29 tokens; $s_8$ is the carry out, and the 9-digit sum is written including a leading 0. The model is a causal language model over the 29 tokens. The loss and every metric are computed on the 9 answer digits only.

**Why plain is primary (a change from the draft, which used reversed output).** In reversed output ($s_0$ first, as recommended for learnability by Lee et al. 2023), the carry into column $i$ follows from three tokens the model has already seen: it is 1 iff $a_{i-1} + b_{i-1} - s_{i-1} \ge 9$. Under teacher forcing, and under greedy decoding once earlier digits are right, every output digit is then a local function of a fixed number of visible tokens, whatever the chain length. A long chain stops being a sequential demand, and $c$ would measure little beyond the frequency of 9-sum columns. In plain format, $s_8$ is written first and depends on the whole chain from the bottom column up. The demand that $c$ is meant to measure is present at prediction time.

**Reversed-format control.** The middle model size is also trained on the reversed format `$ a7…a0 + b7…b0 = s0 s1 … s8 $` over the full $c \times \alpha \times$ seed grid. That tests the mechanism (Gate G7).

**Pre-declared fallback.** If the GPU pilot (Section 2.8) shows that plain format is not learned at this scale (best 6M pilot run above 2.0 bits per answer at $T = 2^{22}$ on $c = 3$), the two formats swap roles. Reversed becomes primary at all three sizes, plain becomes the 6M control, and the report says so prominently.

### 2.3 Exact chain-length strata and the within-stratum sampler (16b)

Let $D[i][\text{carry}][\text{run}][\text{max}]$ count the completions of columns $i..7$, given the incoming carry, the current run length and the longest run so far. Over 8 columns the state space is under $9 \cdot 2 \cdot 9 \cdot 9$, and each column has 100 digit pairs. The exact natural shares (fraction of all $10^{16}$ ordered pairs) are:

| $c$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| share | 0.84% | 19.56% | 33.41% | 23.31% | 12.35% | 5.87% | 2.72% | 1.25% | 0.69% |

(Computed with this dynamic program on 10-18-2026; the counts sum to exactly $10^{16}$. For 3 digits the program matches brute force.)

**Sampler.** To draw a pair uniformly within stratum $c$, walk the columns from least significant to most significant. At each column choose the digit pair $(a_i, b_i)$ with probability proportional to the number of completions that end with longest run exactly $c$. This is exact uniform sampling, by the standard counting argument. Tests:

- for widths 1–4, the sampler's empirical distribution matches brute-force enumeration (chi-square $p \gt 0.001$ on $10^6$ draws per stratum, and exact counts);
- every drawn pair has the requested $c$.

**Strata roles.** Training levels $c \in \{0, 1, 2, 4, 7\}$ (law variable $s \in \{1, 2, 3, 5, 8\}$). Probe strata $c \in \{3, 5, 6, 8\}$ are evaluated but never trained on. They show how well the carry rule transfers, are reported, and are not gated.

### 2.4 Alignment construction, test set and controls (16b)

- **Fresh data.** Every training example is new. Example $j$ of stratum $c$ in training seed $s$ comes from NumPy PCG64 seeded by ($s$, $c$, $j$), so it is identical in every run of that seed (common random numbers across $\alpha$ and model sizes).
- **Schedule.** Example $k$ is a target example iff $\lfloor (k+1)\alpha \rfloor - \lfloor k\alpha \rfloor = 1$, which gives exactly $\lfloor m\alpha \rfloor$ target examples after any $m$. Non-target slots cycle through the other 4 training levels in a seeded permutation, so each gets $(1-\alpha)/4$. Grid: $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$.
- **Test set.** 2,000 pairs per stratum $c = 0, \dots, 8$ (18,000 in all), drawn with the split seed `20261016` from a separate stream. A pair and its commuted twin $(b, a)$ are both excluded from training: the sampler rejects them (the expected rejection rate is below $10^{-10}$ in every stratum; every rejection is counted and logged).
- **Held fixed.** Input length (zero padding), output length (9 digits), the format, and the token positions. There is no task token: the stratum is never signalled to the model.
- **Not held fixed (logged covariates).** Longer chains force more 9-sum and carry-generating columns, so the input-digit and answer-digit distributions differ by stratum. We log per stratum the input-digit total-variation distance from the natural distribution, the answer-digit entropy, the number of carries (as distinct from the longest run), and the training exposure to chains of length $\ge k$ for every $k$ (an example with $c = 7$ also contains every shorter chain).

### 2.5 Models and optimizer (16b)

nanoGPT-style decoder: pre-LayerNorm, learned positions (block size 32), GELU MLP of width $4d$, tied embeddings, no dropout.

| Name | Layers | $d$ | Heads | Total parameters | Non-embedding $N$ | Learning rate |
|---|---|---|---|---|---|---|
| 1M | 5 | 128 | 4 | 997,376 | 991,616 | $2\times10^{-3}$ |
| 6M | 8 | 256 | 8 | 6,330,112 | 6,318,592 | $1\times10^{-3}$ |
| 25M | 8 | 512 | 8 | 25,243,136 | 25,220,096 | $5\times10^{-4}$ |

Counts are from the architecture formula, $12d^2 + 13d$ per layer plus the final LayerNorm. $N$ excludes the token and position embeddings ([Kaplan et al. 2020, arXiv 2001.08361](https://arxiv.org/abs/2001.08361)). The code must assert both counts.

AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1 on matrices, gradient clipping 1.0, batch 256 examples, warmup 200 steps, then a constant learning rate (so every checkpoint lies on one run's learning curve). Training uses bfloat16 autocast. Evaluation uses float64 with deterministic algorithms. Checkpoints at $T \in \{2^{16}, 2^{18}, 2^{20}, 2^{22}, 2^{24}\}$ examples (steps 256 to 65,536).

### 2.6 Seeds, grid and held-out cells (16b)

- **Seeds.** Training seeds `16101, 16202, 16303` (3 seeds; compute). GPU pilot seed `16919`. 16a pilot seed `16909`. Task-sampling and split seed `20261016`. Analysis seed `29101626`. Bootstrap draws 20,000.
- **Primary grid (plain format).** 5 levels × 4 $\alpha$ × 3 sizes × 3 seeds = **180 runs**.
- **Format control (reversed, 6M only).** 5 × 4 × 3 = **60 runs**.
- **Total 240 shards.** Shard id $= 60\,i_N + 12\,i_c + 3\,i_\alpha + i_{\text{seed}}$ for the primary grid ($i_N$ = 0, 1, 2 for 1M, 6M, 25M: shards 0–179). Shard id $= 180 + 12\,i_c + 3\,i_\alpha + i_{\text{seed}}$ for the control (shards 180–239).

| Coordinate | Fit | Held out |
|---|---|---|
| $c$ ($s$) | 0, 1, 2, 7 (1, 2, 3, 8) | 4 (5) |
| $\alpha$ | 0.05, 0.4, 1.0 | 0.15 |
| $T$ | $2^{16}$ to $2^{22}$ | $2^{24}$ |
| $N$ | 1M, 6M, 25M | none |

**Why hold out $c = 4$ rather than the middle index ($c = 2$).** The fit levels then span the whole range $s = 1$–$8$, so the held-out level is interpolated in $\log s$. And $s = 5$ sits in the widest log-gap of the fit levels (between $\log_2 3$ and $\log_2 8$), which is where the coefficient and exponent pictures disagree most for a given fit.

### 2.7 Measurements per run and checkpoint

- **Primary:** $\varepsilon$ = mean over the target stratum's 2,000 test pairs of the summed teacher-forced code length of the 9 answer digits, in bits per answer. Addition is deterministic, so the floor is $L^\star = 0$ and $\varepsilon$ is the loss itself. 16a uses the same definition over its 4 answer digits, on the target stratum's test pairs (1,440 for isolated, 720 otherwise).
- Greedy exact-match accuracy on the target stratum.
- $\varepsilon$ and accuracy on every other stratum, including the probes.
- Per-answer-digit code lengths (which digit carries the excess).
- Tokens per second.

### 2.8 GPU pilot (16b)

Seed `16919`, target $c = 3$ (a probe level, not a grid level) at $\alpha = 0.4$, plain format. Learning-rate multipliers $\{0.5, 1, 2\}$ × the table value for each of the 3 sizes, trained to $T = 2^{22}$: 9 jobs, about 2 A100-hours. Per size, the multiplier with the lowest pilot $\varepsilon$ at $2^{22}$ is used; ties within 0.01 bits per answer go to 1. The reversed control uses the 6M choice. The fallback rule of Section 2.2 is applied to the same pilot.

This selection rule is frozen. The coding agent has no GPU, so the pilot runs after the freeze. Its outcome is recorded outside Sections 2–5: in `config.json` keys `transformer.lr_selected` and `transformer.primary_format_final`, and in a "GPU pilot disclosure" paragraph in Section 7, in one commit made before any GPU shard runs.

## 3. Measurements

Written to `metrics.csv` with columns `seed, target_id, K, K_bits, alpha, T, step, excess_bits, accuracy, n_test`, plus `arm, format, N, c, s, eval_stratum, digit_bits_0..8`. To keep the shared contract, `target_id` is the stratum name (`c0` … `c8`; for 16a `nocarry`, `isolated`, `cascade`), `K` holds $s$, and `K_bits` is empty: there is no description length for a chain length, and the report says so.

Also recorded:

- per mix: the realized stratum counts, the column-carry rates (16a), and the input-digit total-variation distance (16b);
- the exposure-to-chains-$\ge k$ table;
- wall time and throughput.

## 4. Laws and analysis plan

### 4.1 Laws

Floors are 0, so the forms are the program's, with $s$ in place of $K$ and the $N$ term added because $N$ varies in both arms. With $\tilde s = s/s_{\text{ref}}$ (geometric mean of the fit levels), $\tilde T = T/T_0$, $\tilde N = N/N_0$ (smallest size), and $\gamma(s) = \gamma_0 + \gamma_1\log_2\tilde s$:

| Name | Forecast $\hat\varepsilon$ |
|---|---|
| `M0_scale` | $B\tilde N^{-\beta} + A\,\tilde T^{-\gamma}$ |
| `M_align_eff` | $B\tilde N^{-\beta} + A\,(\alpha^\eta\tilde T)^{-\gamma}$ |
| `M_align_add` | $a + b\tilde N^{-\beta} + c\,\tilde T^{-\gamma} - d\,\alpha$ (the alignment-scaling paper's Eq. 1 form) |
| `M_K_coef` | $B\tilde N^{-\beta} + A\,\tilde s^{\kappa}(\alpha^\eta\tilde T)^{-\gamma}$ |
| `M_K_exp` | $B\tilde N^{-\beta} + A\,(\alpha^\eta\tilde T)^{-\gamma(s)}$ |
| `M_K_both` | $B\tilde N^{-\beta} + A\,\tilde s^{\kappa}(\alpha^\eta\tilde T)^{-\gamma(s)}$ |

**Fitting.**

- Per seed, on fit rows only. 16b uses the primary-format rows, pooled over the three sizes. 16a pools its three fit widths.
- Nonlinear least squares on $\log_2\varepsilon$ (`scipy.optimize.least_squares`, trust-region reflective), 64-start grid over exponents.
- Bounds: $\gamma, \gamma_0, \beta \in (0, 3]$; $\gamma_1 \in [-1, 1]$ with $\gamma(s) \gt 0$ on the grid; $\eta \in [0, 3]$; $\kappa \in [-3, 3]$; $A, B \gt 0$.
- `M_align_add` is fit in linear space and can forecast $\le 0$, which counts as invalid.
- `independent_fit.py` (L-BFGS-B on the same objective, a different start grid) must agree on every held-out RMSE (root-mean-square error) to $10^{-4}$ relative.
- **Sensitivity (reported, not gated):** $s = c$ on $c \ge 1$ only; $s$ = number of carries; per-size fits; 16b fits with exact-match error in place of $\varepsilon$.

### 4.2 Held-out scoring and uncertainty

- **Scoring.** RMSE of $\log_2\hat\varepsilon - \log_2\varepsilon$ on held-out rows. Every row with any held-out coordinate is held out, and forecasts are written to `forecasts.csv` and hashed before held-out outcomes are read. Relative reduction is $1 - \mathrm{RMSE}_{\text{new}}/\mathrm{RMSE}_{\text{base}}$.
- **Intervals.** 95% paired seed-block bootstrap, 20,000 draws, analysis seed `29101626`.
- **16b has 3 seeds,** so the bootstrap has only 10 distinct resamples. Every 16b gate therefore also requires each seed's own point estimate to fall on the supported side.
- **16a** has 8 seeds and uses the bootstrap alone.

### 4.3 Data multipliers

$T^\star(s, \alpha)$ is the $T$ at which the fitted learning term reaches $\varepsilon^\star$ at the largest model (25M for 16b; width 128 for 16a). The capacity term $B\tilde N^{-\beta}$ is excluded, so the multipliers compare data bills. The quantities are:

- the chain multiplier $M_s = T^\star(s_{\max})/T^\star(s_{\min})$ over the fit levels;
- the alignment range multiplier $M_\alpha = (\alpha_{\max}/\alpha_{\min})^\eta$ (base ratio $\alpha_{\max}/\alpha_{\min}$ = 20 for 16b, 9 for 16a);
- the elasticities $e_\alpha = \eta$ and $e_s = \partial\log T^\star/\partial\log s$;
- $\varepsilon_{\text{flip}}$ if the exponent picture wins.

These come from `M_K_both`, falling back to `M_K_coef` if more than 25% of seed fits fail to converge (disclosed). A model-free log-log interpolation of each cell's observed curve is reported alongside.

### 4.4 Target excess $\varepsilon^\star$

- **16b:** $\varepsilon^\star = 0.1$ bits per answer, fixed at the freeze. That is about 0.3% of the code length of a uniform guess over the 9 answer digits ($9\log_2 10 \approx 29.9$ bits), low enough that reaching it means the stratum is essentially solved. The GPU pilot must confirm that it is at least 10 times the evaluation standard error over 2,000 test pairs; if not, the report says so.
- **16a:** $\varepsilon^\star = 0.25$ bits per answer, with a pilot rule. If the cascade target at $\alpha = 0.1$, width 64, does not reach $2\varepsilon^\star$ by 1,000 updates, $\varepsilon^\star$ is raised to its value there, rounded up to 0.05, and recorded before the freeze.

Multiplier curves over $\varepsilon \in \{0.03, 0.1, 0.3, 1\}$ are reported regardless.

### 4.5 Mechanism analyses (reported, not gated)

- Per-answer-digit excess. In plain format the excess should sit on the high-order digits ($s_8$ first), and in reversed format it should be flat.
- Fitted $\gamma$ per level against the training exposure to chains $\ge c$ (quanta mechanism).
- Probe-stratum $\varepsilon$ against $c$ (does the carry rule transfer to untrained chain lengths?).
- The 16a units- and tens-carry rate as a covariate, against the 16a $\alpha$ effect.

## 5. Prospective pass/fail gates

Every gate ends `SUPPORTED` or `NOT_ESTABLISHED`, reported with equal prominence. Primary: G1–G4 on 16b in its primary format. Secondary: G5–G7 and the 16a gates. G1–G4 mirror the Experiment 11 gate family. If Experiment 11's frozen protocol defines them differently, the running agent aligns them before this freeze and records it.

1. **G1 — Chain length is a real data cost (C1).** `SUPPORTED` iff $M_s \gt 1$ with 95% lower bound $\gt 1$ at $\varepsilon^\star$, and every seed's $M_s \gt 1$.
2. **G2 — Chain length adds held-out value beyond alignment.**
   - Two comparisons: `M_K_coef` vs `M_align_eff` and `M_K_exp` vs `M_align_eff`.
   - `SUPPORTED` iff at least one gives a held-out relative RMSE reduction $\ge 10\%$, with a Bonferroni-corrected lower bound (1.25th percentile) $\gt 0\%$.
   - The winning law must have 0 invalid forecasts and a positive reduction on the held-out level's rows alone.
3. **G3 — Coefficient or exponent (C3).** Let $\rho = \mathrm{RMSE}(\texttt{M\_K\_exp})/\mathrm{RMSE}(\texttt{M\_K\_coef})$.
   - **Exponent** `SUPPORTED` iff $\rho \le 0.90$ with 95% upper bound $\lt 1$, and $\gamma_1 \lt 0$ in `M_K_both` with 95% upper bound $\lt 0$.
   - **Coefficient** `SUPPORTED` iff $1/\rho \le 0.90$ with 95% upper bound $\lt 1$, and the 95% interval of $\gamma_1$ contains 0 or lies within $[-0.02, 0.02]$.
   - Otherwise `NOT_ESTABLISHED`.
4. **G4 — Alignment dominates once trained (C2).** Both parts are required:
   - (a) $R = \log M_\alpha/\log M_s \gt 1$ with 95% lower bound $\gt 1$;
   - (b) at $\alpha = 1$, the observed gap $\Delta(T) = \varepsilon(c{=}7) - \varepsilon(c{=}0)$, averaged over sizes, satisfies $\Delta(2^{22})/\Delta(2^{16}) \lt 1$ with 95% upper bound $\lt 1$.
5. **G5 (secondary) — Same verdict as exact complexity.**
   - `SUPPORTED` iff G3 here reaches a decided verdict (coefficient or exponent) that equals Experiment 11's decided G3 verdict, and equals Experiment 12's if decided there.
   - The sign of $\gamma_1$ (its 95% interval excluding 0, or contained in $[-0.02, 0.02]$) must be consistent with theirs.
   - `NOT_ESTABLISHED` with the reason named ("Experiment 11 undecided", "Experiment 11 not complete", "verdicts differ") otherwise. A difference is a finding, not a failure of this experiment.
6. **G6 (secondary) — Bigger models shrink the chain multiplier.** Per-size `M_K_both` fits. `SUPPORTED` iff $M_s(25\text{M})/M_s(1\text{M}) \le 0.67$ with 95% upper bound $\lt 1$, the point estimates are monotone in $N$, and every seed agrees in direction.
7. **G7 (secondary) — Chain length is a demand only when the format makes it one.** At 6M, `SUPPORTED` iff $\log M_s(\text{plain})/\log M_s(\text{reversed}) \ge 1.5$ with 95% lower bound $\gt 1$, and every seed agrees. If the formats swapped under the Section 2.2 fallback, the same ratio is still plain over reversed.
8. **16a-G1 to 16a-G4 (secondary).** G1–G4 as defined above, on 16a. Only the bootstrap rule applies (8 seeds). Levels are no-carry ($s = 1$) and cascade ($s = 3$) for the multiplier. G4(b) compares 1,000 against 100 updates.

## 6. What each outcome would mean

| Outcome | Interpretation |
|---|---|
| G1, G2, G4 `SUPPORTED` | A structural demand behaves like a real data cost that alignment outweighs across this grid, as Brando's claim says for complexity. |
| G3 here equals G3 in Experiments 11/12 (G5) | Structural demand and exact complexity enter the law the same way. In-distribution, chain length is a usable proxy for the cost that complexity creates (not for $K$ itself). |
| G3 differs from Experiments 11/12 | Demand and complexity are different kinds of cost. Laws fit on carry strata should not be read as complexity laws. |
| G7 `SUPPORTED` | The chain cost comes from sequential resolution forced by the output format. This also explains why reversed format is easier, as Lee et al. 2023 found. |
| G7 not supported, G1 `SUPPORTED` | Chain length costs data even when locally computable: the cost is the rarity of the column patterns (input-distribution shift), not sequential depth. Check the input-digit total-variation covariate. |
| G1 `NOT_ESTABLISHED` in 16b, `SUPPORTED` in 16a | The earlier carry-difficulty effects need the repeated-epoch, small-MLP regime, or the answer and marginal structure of the 3-digit set. |
| G4 (a) fails | At this scale, chain length moves the data bill at least as much as alignment across the grid, against Brando's claim for this variable. |
| G6 `SUPPORTED` | Scale buys back the chain cost. A $s \times N$ interaction belongs in the full law. |
| The fallback swapped formats | Plain format is not learnable at these sizes. The primary test is then the weaker reversed format, and the report must lead with that. |
| Confound: probe strata transfer poorly | Training on $\{0,1,2,4,7\}$ does not teach a general carry rule. The fitted law is about the trained strata only. |

## 7. Compute estimate and hardware

**16a (CPU).** Measured on 10-18-2026 on the 4-core box (PyTorch 2.14 CPU build, 1 thread): the width-32 to width-128 MLPs at batch 256 take 1.7–3.5 s per 1,000 updates. A run (3,000 updates plus 4 evaluations on 2,880 pairs) takes about 8–15 s, so 384 runs on 4 workers take **about 20–25 min** and the pilot about 1 min. Hard cap `max_wall_seconds = 5400`. RAM under 1 GB. Artifacts about 0.1 GB. 16a can also run as one CPU job on Sherlock's `normal` partition, for example `sbatch --partition=normal --cpus-per-task=4 --mem=16G --time=02:00:00 --wrap '<venv python> expt_v1/run.py --arm mlp --output expt_v1/results/mlp --workers 4'`.

**16b (GPU).** Each example is 29 tokens, so $T = 2^{24}$ examples is $4.9\times10^8$ tokens. Steps of 8,192 tokens are small, so throughput is bound by kernel-launch overhead, not by FLOPs. The estimates below were not benchmarked on GPU.

| Size | 6NT FLOPs at $2^{24}$ | Hours per run (A100, estimate) | Runs | A100-hours |
|---|---|---|---|---|
| 1M | $2.9\times10^{15}$ | 0.15 | 60 | 9 |
| 6M | $1.8\times10^{16}$ | 0.3 | 60 + 60 control | 36 |
| 25M | $7.4\times10^{16}$ | 0.7 | 60 | 42 |
| Pilot, evaluation | | | 9 | about 3 |
| **Total** | | | **240 + 9** | **about 90** |

That is about 12 h on one 8-GPU node, or about 6 h with Sherlock's 16-GPU cap, if GPUs are free. Every listed GPU fits all sizes in memory (25M at 8,192 tokens per step needs about 3 GB), including mercury's 16 GB RTX A4000. The A4000 should be used only for 1M and 6M shards. Data is generated on the fly by 8 CPU workers per GPU using the exact sampler. Checkpoint storage (float32, 5 per run) is about 1.2 GB (1M) + 15 GB (6M with control) + 30 GB (25M) ≈ **47 GB**. Checkpoints stay on cluster storage, and only their SHA-256s enter the repository. Shards use atomic claim directories and DONE markers, exactly as in Experiment 13. Launch scripts are in [`launch/`](launch/README.md).

## 8. Artifacts and verification

Under `expt_v1/results/mlp/` and `expt_v1/results/transformer/`:

- `tasks.json`: one entry per stratum with `task_id`, `family: "addition_carry"`, `role` (target, distractor, probe), `K` = $s$, `K_bits: null`, `definition` (digits, chain-length rule, format), natural share, counts, and covariates.
- `data/<task_id>_test.npz`: exact test pairs.
- `data/<task_id>_train.npz`: 16a only. 16b streams are regenerated from seeds, and `mixes.json` holds the counts plus a SHA-256 of each run's stream.
- `metrics.csv` (Section 3).
- `checkpoints/<seed>_<target_id>_<alpha>[_<size>_<format>]/T_<T>.npz`: float32 numpy arrays.
- `fits.json`, `forecasts.csv`, `gates.json`, `bootstrap.npz`, `manifest.json`, `ARTIFACT_SHA256.json`.

`verify.py` prints PASS only if all of these hold:

1. Reloading every checkpoint and re-evaluating in float64 reproduces every stored $\varepsilon$ to `max_abs_diff < 1e-9` bits.
2. Every test pair's sum and $c$ are recomputed with an independent column-addition oracle (Experiment 06's `column_addition`).
3. The 16b stratum counts are recomputed by brute force at 3 digits and by the dynamic program at 8 digits (total $10^{16}$).
4. Training streams are regenerated from seeds, and the hashes match.
5. Train/test disjointness holds, including commuted twins.
6. `independent_fit.py` agrees.

## 9. Scope and limits

- Chain length is a structural demand, not Kolmogorov complexity or $K_{\mathcal L}$. The results say how a demand variable enters the law, not how complexity does.
- 16a reuses a small set over many epochs and spans only $c \le 2$. Its column-carry marginals differ by target (disclosed numbers above).
- 16b strata differ in input-digit statistics by construction (Section 2.4). These are logged, not removed.
- Three seeds in 16b. No held-out model size.
- One format is primary. The other is tested only at 6M.
- A constant learning rate.
- Teacher-forced bits measure the predictive distribution. Exact match is reported but not gated.
- The results concern these toys and budgets. They do not show that benchmark difficulty in large language models is mostly alignment.

**TLDR-end:** [tdc: Expt 16 design] Cross carry-chain length (a structural demand, not $K$) with target-stratum share on fresh 8-digit addition for three transformer sizes, plus a 3-digit MLP arm reusing the Experiment 06/08 data, fit the program's law family with held-out chain level, share, $T$ and width, and test whether chain length costs data, whether alignment dominates, whether it moves the coefficient or the exponent as exact complexity does in Experiments 11/12, and whether the cost exists only when the output format makes the chain sequential.
