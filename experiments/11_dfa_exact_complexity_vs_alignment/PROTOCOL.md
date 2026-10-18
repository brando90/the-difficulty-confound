# Protocol: Exact DFA complexity versus data alignment on regular languages (`11_dfa_exact_complexity_vs_alignment`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/11_dfa_exact_complexity_vs_alignment/PROTOCOL.md>

Status: Frozen on 10-18-2026 before any gated Experiment 11 training. (Drafted 10-18-2026 from Brando Miranda's complexity-versus-alignment idea; the only pre-freeze changes are the pilot outcomes recorded in Section 2.6. After this freeze nothing in Sections 2–5 changes.)

Notation follows the [program overview](../ideas/complexity_vs_alignment/README.md): $\varepsilon$ is held-out excess loss in bits, $T$ is training examples seen, $\alpha$ is the target's share of the training mix, $K$ is exact complexity in a stated description language, and the laws are `M0_scale`, `M_align_eff`, `M_align_add`, `M_K_coef`, `M_K_exp` and `M_K_both` with parameters $A, \eta, \kappa, \gamma, \gamma_0, \gamma_1$.

## 1. Scientific motivation

**Hypothesis.** Brando (10-18-2026): "kolmogorov complexity would mean that it does take more 'effort' to train, but if you already trained on it the models will perform better. So it's not that complexity doesn't exist but data alignment dominates." Experiments 01–09 could not test this, because addition's complexity never changed: carry-chain length is a structural demand, not Kolmogorov complexity ([COMPLEXITY.md](../archive/01_fixed_width_addition_exposure/COMPLEXITY.md), [predictability bridge](../ideas/predictability_bridge/README.md#where-kolmogorov-complexity-belongs)). Here we need a family of tasks whose complexity we can compute exactly and set on purpose, crossed with alignment that we also set on purpose.

**Why regular languages.** A regular language has a unique minimal deterministic finite automaton (DFA), up to renaming states (the Myhill–Nerode theorem; Sipser, *Introduction to the Theory of Computation*, ch. 1). Its size is computable exactly with Hopcroft's algorithm ([Hopcroft 1971, Stanford CS-TR-71-190](http://i.stanford.edu/pub/cstr/reports/cs/tr/71/190/CS-TR-71-190.pdf)). We fix a description language $\mathcal L_{\text{DFA}}$: the start state is state 0, and each of the $n$ states is written as two transitions of $\lceil \log_2 n \rceil$ bits each plus one accept bit. That gives

```math
K_{\text{bits}}(n) = n\,\big(2\lceil \log_2 n\rceil + 1\big), \qquad K_{\text{bits}} = 20,\ 56,\ 144,\ 352 \ \text{ for } n = 4, 8, 16, 32 .
```

Every DFA for the language has at least $n$ states and this code length increases with $n$, so $K_{\text{bits}}$ is the exact shortest description of the language in $\mathcal L_{\text{DFA}}$. It is not universal Kolmogorov complexity. By the invariance theorem it upper-bounds universal $K$ up to an additive constant, and every claim below is stated for this language only.

**Claims tested** (from the overview):

- **C1, complexity is real.** More DFA states cost more data to reach the same held-out loss.
- **C2, alignment dominates once trained.** In units of data, changing the target share moves the bill more than changing complexity does, and at $\alpha = 1$ the complexity gap in loss shrinks as $T$ grows.
- **C3, which law.** Does complexity move the coefficient (minimum description length (MDL) picture, `M_K_coef`: $\kappa \gt 0$, $\gamma_1 = 0$) or the exponent (quanta picture, `M_K_exp`: $\gamma_1 \lt 0$)?

**Competing predictions.**

| Picture | Prediction here |
|---|---|
| MDL / Occam (Grünwald 2007, ch. 8; [Grünwald's MDL tutorial](https://arxiv.org/abs/math/0406077)) | Excess loss scales like $K/T$: $\kappa \gt 0$, $\gamma_1 = 0$. The complexity data multiplier $(K_2/K_1)^{\kappa/\gamma}$ is the same at every accuracy, so if alignment wins at one $\varepsilon^\star$ it wins at all of them. |
| Quanta ([Hutter 2021](https://arxiv.org/abs/2102.04074); [Michaud et al. 2023](https://arxiv.org/abs/2303.13506)) | Bigger automata add rarely visited states, which lengthens the sub-skill frequency tail and lowers the exponent: $\gamma_1 \lt 0$. The complexity multiplier grows as $\varepsilon^\star \to 0$, so alignment dominates only above some accuracy. |
| Sensitivity, not size ([Hahn and Rofin 2024](https://arxiv.org/abs/2402.09963)) | Transformers find high-sensitivity functions hard. For random DFAs, sensitivity rises with $n$ (Section 2.2), so a $K$ effect could be a sensitivity effect. We stratify and measure it. |
| Expressivity limit ([Liu et al. 2022](https://arxiv.org/abs/2210.10749); [Delétang et al. 2022](https://arxiv.org/abs/2207.02098); [Bhattamishra et al. 2020](https://arxiv.org/abs/2009.11264)) | A 2-layer transformer may not be able to represent some large automata at length 24. Then the curve flattens at a floor that no amount of data removes. This would look like a smaller exponent, but the cause is architecture, not data. The optional LSTM (long short-term memory network) arm and the plateau diagnostics separate the two. |

**Why the alignment variable is clean here.** Every example carries a task token, so labels from different tasks never conflict. The target's share $\alpha$ is exactly the token-weighted alignment of [the alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws) (Eq. 2), with score $s(x) = 1$ for target examples and $0$ for distractors. String lengths are drawn independently of the task, so the token-weighted and example-weighted shares agree in expectation. Distractor data can only help through shared parameters (transfer) or hurt by competing for capacity (interference). The exponent $\eta$ in $\alpha^\eta T$ measures which of these happens.

## 2. Experimental design

### 2.1 Inputs, splits and labels

- **Input distribution.** Strings over $\{a, b\}$. The length $\ell$ is uniform on $[8, 24]$, and characters are independent and uniform.
- **Train/test partition by exact string, shared by every task.** A string is in the test partition if and only if the first byte of SHA-256 of its ASCII text is below 32 (one eighth of all strings). Training strings are drawn only from the train partition and test strings only from the test partition. No test string is ever seen in training under any task token. Because the partition is a hash, every length is split in the same 1:7 ratio.
- **Test set.** There is one test set, used for every task: 4,096 strings, with 241 at each length from 8 to 23 and 240 at length 24, drawn uniformly with replacement from the test partition of each length (task seed 20261011). At lengths 8–10 the test partition holds only about 32–128 distinct strings, so the test set repeats them. It is an i.i.d. sample of the input distribution restricted to the test partition.
- **Training pools.** Each of the 32 tasks has its own fixed pool of 4,096 train-partition strings per length (69,632 strings), drawn with a task-specific seed derived from 20261011. Short lengths repeat inside pools, as they do under the input distribution.
- **Label.** Accept or reject. The task is binary classification with a deterministic label, so the Bayes floor is $L^\star = 0$ and $\varepsilon$ is the held-out cross-entropy in bits.

### 2.2 Exact complexity, task sampling and covariates

**Exact $K$.** Hopcroft minimization after removing unreachable states. A sampled DFA is kept only if its minimal DFA has exactly $n$ states. Correctness check (unit test): on at least 1,000 random DFAs with 2–10 states, Hopcroft's partition must equal brute-force Moore partition refinement, and each minimized DFA must be language-equivalent to its source. Equivalence is checked by product-automaton search and also by brute force over all strings of length below $n_1 + n_2$.

**Sampler.** To get a DFA with exactly $n$ minimal states, draw $m = \lceil 1.25 n\rceil + 1$ states with uniform random transitions and Bernoulli(0.5) accept bits, start at state 0, keep the reachable part, minimize, and accept if the result has exactly $n$ states. The accepted DFA is relabeled in breadth-first canonical order. A prototype on 10-18-2026 needed about 4–8 draws per accepted DFA.

**Filters (all exact, computed by dynamic programming over the input distribution):**

1. Overall acceptance rate in $[0.35, 0.65]$.
2. Acceptance rate in $[0.10, 0.90]$ at every length from 8 to 24. This rules out languages that length alone decides, which a model with position embeddings could shortcut.
3. The language differs from, and is not the complement of, every task already chosen.

**Covariates (exact).**

- *Sensitivity:* the mean over inputs of the fraction of single-character substitutions that flip the label. It is computed by propagating (original state, substituted state) pairs. A Monte Carlo check with 20,000 strings must agree within 4 standard errors.
- *State-visitation frequencies:* the expected number of visits to each state over positions $0..\ell$, averaged over $\ell$ and normalized.
- *Visitation tail steepness* $\tau = -b$, where $b$ is the ordinary-least-squares slope of $\log$ frequency on $\log$ rank.
- *Acceptance rate,* overall and by length.

**Sensitivity-stratified selection (design control).** A prototype on 10-18-2026 (40 accepted DFAs per size) gave median sensitivity 0.17, 0.30, 0.36 and 0.42 for $n = 4, 8, 16, 32$. The interquartile ranges of $n = 4$ and $n = 32$ do not overlap. So sensitivity cannot be matched across levels without choosing atypical, parity-like small automata. Instead we spread it within levels. For each $n$, draw candidates (task seed 20261011) until 64 distinct DFAs pass the filters, sort them by sensitivity, and cut them into four quartile bins of 16. From each bin, draw two at random without replacement: the first becomes a **target** and the second a **distractor**. This gives 4 targets and 4 distractors per level, each spanning that level's sensitivity range. The within-level spread is what lets Section 4.6 separate $K$ from sensitivity.

| | $n = 4$ | $n = 8$ | $n = 16$ | $n = 32$ | total |
|---|---|---|---|---|---|
| $K_{\text{bits}}$ | 20 | 56 | 144 | 352 | |
| targets | 4 | 4 | 4 | 4 | 16 |
| distractors | 4 | 4 | 4 | 4 | 16 |
| task tokens | | | | | 32 |

### 2.3 The alignment construction

- **Mix.** For target $j$ and share $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$, a fraction $\alpha$ of examples come from target $j$. The remaining $1-\alpha$ is split equally over the **same 16 distractors in every condition**. So the non-target part of the mix has the same distribution whatever the target's $K$. Targets are never distractors for each other.
- **Exact schedule.** Training runs to $T_{\max} = 2^{16}$ examples. In each octave interval ($[1, 2^9]$, then $(2^{k-1}, 2^k]$ for $k = 10..16$), exactly $\mathrm{round}(\alpha \times \text{interval length})$ slots go to the target, placed uniformly at random. The other slots are dealt to the 16 distractors in equal counts ($\pm 1$). So the target count at every checkpoint is within $\pm 4$ of $\alpha T$. The laws use the nominal $\alpha T$, and `mixes.json` records the exact per-task counts at each checkpoint and the data-order seed.
- **What $T$ counts.** $T$ counts all examples seen, target and distractor. The target sees $\alpha T$ of them. Compute per run is identical across $\alpha$.
- **Batches of one length.** Each batch of 64 has a single length. Lengths come from a balanced schedule (each length gets 60 or 61 of the 1,024 batches), shuffled by the data-order seed. This avoids padding, which cuts compute by about a third, and leaves the length distribution unchanged. At $\alpha = 1$ the most any length uses is $61 \times 64 = 3{,}904$ pool strings, which is below 4,096, so each example is a fresh pool draw. Within a run, a task's pool is read in a seed-specific permutation without replacement.

### 2.4 Model, optimizer and checkpoints

- **Transformer (primary).** Decoder-only with causal attention: 2 layers, $d_{\text{model}} = 64$, 4 heads, multilayer perceptron (MLP) width 256, pre-LayerNorm, GELU, and learned position embeddings for 25 positions. The input is `[task token, c_1, …, c_ℓ]` over a 34-token vocabulary (a, b, 32 task tokens), and the prediction is a 2-way softmax at the last position. It has **104,002 parameters**: embeddings 2,176, positions 1,600, two blocks of 49,984, final LayerNorm 128, head 130. This count was checked by instantiating the model on 10-18-2026, and `run.py --check-only` asserts it.
- **Optimizer.** AdamW, learning rate $10^{-3}$, weight decay 0.01, betas (0.9, 0.999), batch 64, gradient-norm clip 1.0. There is a linear warmup over the first 8 steps, after which the rate stays constant. With batch 64, 8 steps is exactly the first checkpoint ($T = 2^9$), so every checkpoint is taken at the same constant rate. (A 100-step warmup would end at $T = 6{,}400$ and leave 4 of the 8 checkpoints inside warmup.) The learning rate is not tuned per condition.
- **Checkpoints.** $T \in \{2^9, 2^{10}, \dots, 2^{16}\}$, which is steps 8 to 1,024. These are 8 checkpoints along one trajectory. This is an "anytime" learning curve at constant rate, not a set of separately tuned runs per $T$. The one-seed pilot (Section 2.6) may move this window under fixed rules.
- **Optional LSTM arm (exploratory, no gate).** A one-layer LSTM with embedding 32 and hidden size 128, reading the same tokens, with a 2-way head on the last hidden state. It has 84,290 parameters (instantiated 10-18-2026). It runs at $\alpha = 1$ for all 16 targets and 8 seeds (128 runs), with the same optimizer and checkpoints, only if at least 45 minutes remain under the wall cap after the main grid. An LSTM can simulate any DFA exactly, so it tells us whether a transformer plateau is an expressivity limit.

### 2.5 Seeds, grid and splits

- **Training seeds** 11101, 11202, 11303, 11404, 11505, 11606, 11707, 11808. The seed sets the initialization and the data order (schedule, batch lengths, pool permutations). The tasks, pools and test set are fixed by task seed 20261011 and shared by all seeds. Analysis seed 29101126; 20,000 bootstrap draws.
- **Runs.** 4 levels × 4 targets × 4 $\alpha$ × 8 seeds = **512 runs**, each 1,024 steps. That makes 4,096 checkpoint rows and 512 × 8 × 4,096 = 16.8 million held-out predictions.
- **Fit and held-out cells.**

| Axis | Fit | Held-out |
|---|---|---|
| $K$ (states) | 4, 8, 32 | 16 (interpolation) |
| $\alpha$ | 0.05, 0.4, 1.0 | 0.15 (interpolation) |
| $T$ | $2^9 … 2^{15}$ | $2^{16}$ (extrapolation) |

A point is (seed, target, $\alpha$, $T$). Per seed there are 512 points: 252 fit points (12 targets × 3 $\alpha$ × 7 $T$) and **260 held-out points**, the union of the $K = 16$, $\alpha = 0.15$ and $T = 2^{16}$ slices. Over 8 seeds that is 2,080 held-out points.

### 2.6 Calibration pilot (disclosed; seed 11909, never reused)

Run one seed at $\alpha = 1$ for all 16 targets to $T = 2^{17}$: 16 runs, about 11 minutes. Before the freeze, apply these rules mechanically and record them in `CKPT`:

- **P1.** If the median over targets of $T^\star(0.1\ \text{bits})$ is above $2^{14}$, the primary $\varepsilon^\star$ becomes 0.3 bits, and 0.1 and 0.03 bits become the secondary levels.
- **P2.** If more than 4 of the 16 targets are already at or below 0.1 bits at $T = 2^{10}$, shift the whole window down one octave, to $2^8 … 2^{15}$.
- **P3.** Report the measured step time. If the projected full run exceeds 90% of the wall cap, stop and ask Brando. Do not cut seeds silently.

The pilot also reports how many targets improve between $2^{16}$ and $2^{17}$. That shows what the budget leaves out. Pilot outcomes are never pooled with the gated runs.

**Pilot disclosure (10-18-2026, before the freeze).** The pilot ran on the CPU box (Intel Xeon 2.20 GHz, 2 physical cores with 4 hardware threads, shared with other jobs) at learning rate $10^{-3}$ with seed 11909: 16 runs, 959 s, mean 237 s per 2,048-step run under 4 concurrent workers.

- **Learning was slow.** At $\alpha = 1$ and $T = 2^{16}$ the held-out loss ranged from 0.31 to 0.99 bits (median 0.90). No target reached 0.1 bits even at $2^{17}$; at $2^{17}$ the best three were 0.14, 0.20 and 0.24 bits (`dfa_n04_t2`, `dfa_n08_t1`, `dfa_n04_t1`). The $n = 16$ and $n = 32$ targets stayed at 0.69–0.99 bits. Per-target losses are in `expt_v1/pilot_summary.json`.
- **P1 fired** (median $T^\star(0.1) \gt 2^{14}$; it is right-censored at $2^{17}$). **P2 did not fire** (0 of 16 targets at or below 0.1 bits at $2^{10}$). **P3 did not fire**: projected grid 15,178 s, below 90% of the cap (16,200 s). 15 of 16 targets improved from $2^{16}$ to $2^{17}$, and 7 by more than 10%, so the $2^{16}$ budget leaves most of the learning curve out.
- **Learning-rate calibration (added before the freeze, rule written down before its results).** Because the gated grid would have been right-censored almost everywhere, one calibration ran on a fresh seed, 11910 (never reused), at $\alpha = 1$ for all 16 targets over the gated window $2^9 … 2^{16}$, at learning rates $3 \times 10^{-3}$ and $10^{-2}$, with no checkpoints saved. Rule: keep the learning rate with the lowest median over targets of $\varepsilon(2^{16})$, with $10^{-3}$ read from the pilot. Medians were 0.899 bits ($10^{-3}$), 0.931 ($3 \times 10^{-3}$) and 0.977 ($10^{-2}$). **The learning rate stays $10^{-3}$.** Outputs are in `expt_v1/results_calib_lr0.003/` and `results_calib_lr0.01/`, never pooled.
- **New rule P1b (written down before the calibration results).** If the median $T^\star(0.3\ \text{bits})$ is also above $2^{14}$, the primary $\varepsilon^\star$ becomes 0.5 bits and the levels become $\{0.5, 0.3, 0.1\}$. It fired: the median $T^\star(0.3)$ is right-censored at $2^{16}$ at every learning rate tried.
- **Consequences, fixed now.** The primary $\varepsilon^\star$ is **0.5 bits** wherever Sections 4–5 say "0.1 bits (or the P1 level)", and the $\varepsilon^\star$ levels are $\{0.5, 0.3, 0.1\}$. G4's model-free fallback moves the same way: if more than 50% of (target, $\alpha$) pairs are censored at 0.5 bits in the median seed, $E_{\text{mf}}$ is evaluated at **0.7 bits**. The Section 4.6 regression uses the primary level (0.5 bits). The window stays $2^9 … 2^{16}$, and the seeds, grid, learning rate, model and gate thresholds are unchanged.
- **What this predicts.** Many gated runs, especially at $n \ge 16$ and $\alpha \lt 1$, will not reach even 0.5 bits by $2^{16}$, so $T^\star$-based statistics (G1, $E_{\text{mf}}$, the Section 4.6 regression) will be heavily right-censored. Censored values enter as $T_{\max}$, which biases $M_K$ toward 1 (Section 4.4). The law fits on $\varepsilon$ (G2, G3, G5–G7) do not depend on $T^\star$.

## 3. Measurements

Per checkpoint, written to `metrics.csv` with one row per (seed, target, $\alpha$, $T$):

- `excess_bits` $= \varepsilon$: the mean over the 4,096 test strings of $-\log_2 p_\theta(\text{correct label})$, computed in float32 and accumulated in float64.
- `accuracy`: greedy accuracy. `n_test` = 4,096.
- `excess_bits_by_length`: 17 values.
- `target_examples_seen` and `distractor_examples_seen`: exact counts.
- `step`, `T`, wall time.
- Per-string bits for every checkpoint, saved to `predictions/<seed>_<target>_<alpha>.npz` (shape 8 × 4,096, float32).

**Derived per run:** $T^\star(\varepsilon^\star)$ for $\varepsilon^\star \in \{0.3, 0.1, 0.03\}$ bits. Take the first checkpoint with $\varepsilon \le \varepsilon^\star$ and interpolate $\log_2 \varepsilon$ linearly in $\log_2 T$ from the checkpoint before it.

- *Right-censoring.* If the run never reaches $\varepsilon^\star$, $T^\star$ is right-censored and recorded as $T_{\max}$ with a flag.
- *Left-censoring.* If the first checkpoint is already below $\varepsilon^\star$, $T^\star$ is left-censored and recorded as $T_{\min}$ with a flag.
- *Plateau diagnostic.* A run is flagged if $\varepsilon(2^{16}) \ge 0.8\,\varepsilon(2^{14})$ while $\varepsilon(2^{16}) \gt 0.1$ bits.

**Per task** (in `tasks.json`): $n$, $K_{\text{bits}}$, the definition, acceptance (overall and by length), sensitivity and its quartile, visitation frequencies and $\tau$.

**Resolution floor.** $\varepsilon_{\text{res}} = 10^{-3}$ bits. Below it, held-out loss mostly measures how far confidence has grown, not errors. Points below the floor are never dropped. They enter fits and scores through a censored residual (Section 4.1).

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

- **Primary $K$ regressor:** $K_{\text{bits}}$, the description length in bits, which is the unit of the MDL prediction. Every fit is repeated with $K = n$ (states) and reported alongside.
- **Per-seed fits.** Each law is fit separately per seed on that seed's 252 fit points by nonlinear least squares on $\log_2 \varepsilon$ (SciPy `least_squares`, trust-region reflective). The censored residual is $r = \log_2 \hat\varepsilon - \log_2 \varepsilon$ when $\varepsilon \ge \varepsilon_{\text{res}}$, and $r = \max(0, \log_2 \hat\varepsilon - \log_2 \varepsilon_{\text{res}})$ otherwise.
- **Bounds.** Every exponent $\gamma$, including $\gamma_0 + \gamma_1\log_2 K$ at $K_{\text{bits}} \in [20, 352]$, lies in $[10^{-3}, 3]$; $\eta \in [0, 3]$; $\kappa \in [-5, 5]$.
- **Starts.** 64 starts form a 4 × 4 × 4 grid: $\gamma$ or $\gamma_0 \in \{0.1, 0.3, 0.6, 1.2\}$; $\eta \in \{0.25, 0.75, 1.25, 2.0\}$; and a third exponent axis that is $\kappa \in \{-0.5, 0.5, 1, 2\}$ for `M_K_coef`, $\gamma_1 \in \{-0.15, -0.05, 0, 0.05\}$ for `M_K_exp`, and $(\kappa, \gamma_1) \in \{(0,0), (1,0), (0,-0.1), (1,-0.1)\}$ for `M_K_both`. Laws with fewer exponents use the sub-grid. $\log A$ starts at its least-squares value given the exponents.
- **`M_align_add`** is the first-order form of the alignment-scaling paper with Varun. With $N$ fixed it has no $N$ term, and here $L = \varepsilon$. It is fit by least squares on $\varepsilon$ in linear space, as in that paper, with $a, c, d$ free and $\gamma \in [10^{-3}, 3]$. It is scored on the same $\log_2$ scale as the others. A forecast $\le 0$ counts as invalid and is clipped to $\varepsilon_{\text{res}}$ for root-mean-square error (RMSE) scoring.
- **Independent fitter.** `independent_fit.py` profiles $\log A$ out analytically and minimizes over the exponents with L-BFGS-B from the same starts. Its objective value and held-out forecasts must match `laws.py` to $10^{-4}$ relative before anything is reported.

### 4.2 Held-out scoring

Forecasts for all 260 held-out points per seed are written to `forecasts_frozen.json`, and its SHA-256 is logged in `CKPT`, before any held-out row is read. `analyze.py` reads fit rows and held-out rows in separate stages to enforce this. The score is the RMSE of the censored residual on $\log_2 \varepsilon$, pooled over seeds. Relative reduction of law B over law A is $1 - \text{RMSE}_B/\text{RMSE}_A$. An invalid forecast is one that is nonfinite or $\le 0$.

### 4.3 Uncertainty

All intervals are 95% percentile intervals from a seed-block bootstrap: 20,000 draws that resample the 8 seeds with replacement, using analysis seed 29101126. Points and checkpoints within a seed are never treated as independent. A law parameter's estimate is the mean over seeds of the per-seed estimates. Where the gates say so, a **two-stage bootstrap** additionally resamples the 4 targets within each $K$ level in every draw. It answers whether the result generalizes beyond these particular 16 automata. Seed-only intervals are conditional on the sampled targets, and the report says so.

### 4.4 Data multipliers and elasticities

- **Complexity multiplier (model-free).** At $\alpha = 1$ and $\varepsilon^\star$:
  $M_K = \mathrm{GM}_j\, T^\star_j(n{=}32) \,/\, \mathrm{GM}_j\, T^\star_j(n{=}4)$, where GM is the geometric mean over the 4 targets at that level. It is computed per seed, and the point estimate is the geometric mean over seeds. Right-censored values enter as $T_{\max}$. For $n = 32$ this underestimates $T^\star$, and left-censored $n = 4$ values at $T_{\min}$ overestimate $T^\star$. Both make $M_K$ smaller, which is the conservative direction for C1.
- **Elasticities (law-based).** Fit `M_K_both` per seed on **all** 512 points (a descriptive refit, not a forecast). The data needed is

  ```math
  \log T^\star(K,\alpha) = \eta\log\tfrac{1}{\alpha} + \frac{\log A + \kappa \log K - \log \varepsilon^\star}{\gamma(K)}, \qquad \gamma(K) = \gamma_0 + \gamma_1\log_2 K .
  ```

  The alignment elasticity is $e_\alpha = \partial \log T^\star/\partial \log(1/\alpha) = \eta$. The complexity elasticity is $e_K = \partial \log T^\star/\partial \log K$, evaluated analytically at the grid's geometric centre $K_{\text{bits}} = 86.8$. The ratio is $E = \bar e_\alpha / \bar e_K$, the ratio of seed means. If $\bar e_K \le 0$ in a draw, set $E = +\infty$ for that draw.
- **Units.** $E$ depends on the unit of $K$. In states, $E_{\text{states}} = E_{\text{bits}}/1.38$, where 1.38 is the least-squares slope of $\log K_{\text{bits}}$ on $\log n$ over the four levels. Both are reported, and the gate requires both, so the state unit is the binding one.
- **Model-free check.** $E_{\text{mf}}$: per seed, regress $\log_2 T^\star_{j,\alpha}$ on $\log_2(1/\alpha)$ and $\log_2 K_{\text{bits},j}$ over the (target, $\alpha$) pairs that are not censored. The number of censored pairs is reported.
- **Accuracy dependence.** $E$ is reported at $\varepsilon^\star \in \{0.3, 0.1, 0.03\}$ bits and as a curve over $[10^{-3}, 1]$ bits. Under the exponent picture $E$ falls as $\varepsilon^\star$ falls. The report must give the flip point $\varepsilon_{\text{flip}}$ where $E = 1$, if it lies in that range.
- **Range ratio** (the overview's wording of C2, reported, no gate): $R = [T^\star(\alpha{=}0.05)/T^\star(\alpha{=}1)] \,/\, M_K$. Because $\ln 20 \gt \ln(352/20)$, $R \gt 1$ is a weaker requirement than $E \gt 1$.

### 4.5 Alignment comparison (reported, no gate)

`M_align_eff` versus `M_align_add` on the joint held-out set: RMSE, relative reduction with its interval, and invalid-forecast counts for each. Also $\eta$ with its interval from `M_align_eff` and from the winning $K$ law. $\eta = 1$ means only target examples count; $\eta \lt 1$ means the distractors transfer positively; $\eta \gt 1$ means they interfere.

### 4.6 Mechanism and confound analyses

- **Per-target exponent.** For each target, fit $\varepsilon = A_j T^{-\gamma_j}$ to its $\alpha = 1$ curve, per seed, with the censored residual. $\gamma_j$ is the mean over seeds.
- **Quanta test (G7).** Spearman correlation between $\gamma_j$ and the visitation tail steepness $\tau_j$ across the 16 targets. The quanta picture predicts a positive sign, because a steeper frequency decay means fewer rare sub-skills and faster learning. The partial Spearman controlling for $\log K_{\text{bits}}$ is also reported.
- **$K$ versus sensitivity (reported).** Regress per-target $\log_2 T^\star_j(\alpha{=}1, 0.1\ \text{bits})$, averaged over seeds, on standardized $\log K_{\text{bits}}$ and standardized sensitivity. Report both coefficients with seed-block and two-stage intervals, and the variance inflation factor.
- **Plateau census.** Count plateau-flagged runs by level. If the LSTM arm ran, give its $M_K$ and its plateau count.
- **Per-length loss** at $T = 2^{16}$ by level. This checks whether large automata fail mainly on long strings.

## 5. Prospective pass/fail gates

Every gate ends `SUPPORTED` or `NOT_ESTABLISHED`, and all are reported with equal prominence. Primary gates are G1–G4; secondary gates are G5–G7. All thresholds are fixed at the freeze.

1. **G1, complexity is real (C1, primary).** $M_K$ at $\alpha = 1$ and $\varepsilon^\star = 0.1$ bits (or the P1 level). `SUPPORTED` iff $M_K \ge 2.0$, its seed-block 95% lower bound is $\gt 1.0$, and its two-stage (seeds and targets) 95% lower bound is $\gt 1.0$.
2. **G2, complexity adds predictive value beyond alignment (primary).** Compare the $K$ law against `M_align_eff` on the joint held-out set. The $K$ law is whichever of `M_K_coef` and `M_K_exp` has the lower fit-cell RMSE. Both have 4 parameters, and the choice is made before held-out rows are read. `SUPPORTED` iff the relative RMSE reduction is $\ge 10\%$, its 95% lower bound is $\gt 0\%$, the $K$ law's RMSE is also lower on the $K = 16$ slice alone, and there are 0 invalid forecasts.
3. **G3, which law (C3, primary; two mutually exclusive sub-gates).**
   - **G3a, coefficient (MDL) picture.** `SUPPORTED` iff, in `M_K_both`, $\kappa$'s 95% interval lies above 0 and $\gamma_1$'s interval contains 0, and the held-out RMSE of `M_K_coef` is $\le$ that of `M_K_exp`.
   - **G3b, exponent (quanta) picture.** `SUPPORTED` iff, in `M_K_both`, $\gamma_1$'s 95% interval lies below 0, and `M_K_exp` cuts held-out RMSE relative to `M_K_coef` by $\ge 5\%$ with a 95% lower bound $\gt 0\%$.
   - If both are `NOT_ESTABLISHED`, C3 is reported as **unresolved**. They cannot both be `SUPPORTED`.
4. **G4, alignment dominates in units of data (C2, primary).** `SUPPORTED` iff all three hold:
   - the law-based $E$ at $\varepsilon^\star = 0.1$ bits (or the P1 level) is $\gt 1$ with a seed-block 95% lower bound $\gt 1$;
   - the same holds in both $K$ units;
   - the model-free $E_{\text{mf}}$ point estimate is $\gt 1$. If more than 50% of (target, $\alpha$) pairs are censored at 0.1 bits in the median seed, $E_{\text{mf}}$ is evaluated at 0.3 bits instead, and this fallback is fixed now.
5. **G5, complexity gap shrinks with training (C2, secondary).** At $\alpha = 1$, let $g(T) = \bar\varepsilon_{n=32}(T) - \bar\varepsilon_{n=4}(T)$ in bits, the difference of target means. Use the ratio of seed means $\rho = g(2^{16})/g(2^{12})$. `SUPPORTED` iff $g(2^{12}) \gt 0$, $\rho \le 0.5$, and the 95% upper bound of $\rho$ is $\lt 1.0$. This is a gap in absolute bits, so both pictures can pass it. The gap in $\log_2 \varepsilon$ is reported next to it, and that version is the one that separates the pictures.
6. **G6, alignment sets a floor (secondary).** `M_K_floor` adds $a_1(1-\alpha)$ with $a_1 \ge 0$ to the G2 $K$ law. `SUPPORTED` iff $a_1$'s 95% lower bound is $\gt 0$ and `M_K_floor` cuts joint held-out RMSE relative to the G2 $K$ law by $\ge 10\%$ with a 95% lower bound $\gt 0\%$. The prediction is `NOT_ESTABLISHED`: at this capacity, distractors should change the exchange rate ($\eta$) rather than impose a floor. Either outcome is informative.
7. **G7, quanta mechanism (secondary).** `SUPPORTED` iff Spearman($\gamma_j$, $\tau_j$) over the 16 targets is $\ge 0.5$ with a seed-block 95% lower bound $\gt 0$. The two-stage interval is reported next to it.

## 6. What each outcome would mean

| Outcome | Interpretation |
|---|---|
| G1, G2, G4 and G3a `SUPPORTED` | Brando's hypothesis in its coefficient form. Complexity costs a fixed multiple of data at every accuracy, and alignment's exchange rate is larger, so alignment dominates everywhere in the tested range. |
| G1, G2 and G3b `SUPPORTED` | Complexity is real and grows with accuracy. C2 holds only above $\varepsilon_{\text{flip}}$, and the report must say where it flips. "Alignment dominates" then needs that accuracy qualifier. |
| G1 `NOT_ESTABLISHED` | Minimal-DFA size did not measurably raise the data bill for this learner and budget. Check the sensitivity regression and the plateau census before concluding that complexity is not real here. |
| G1 `SUPPORTED`, but sensitivity absorbs $\log K$ in the Section 4.6 regression | The cost tracks sensitivity, which random automata gain as they grow. $K$ is then a proxy, and C1 should be restated as a sensitivity effect. |
| Many $n = 32$ runs plateau at every $\alpha$, while the LSTM learns them | An expressivity limit of the 2-layer transformer, not a data cost. $\gamma_1 \lt 0$ from such data must not be read as quanta. |
| G4 `NOT_ESTABLISHED` with $E \lt 1$ | Per log unit, complexity costs more data than misalignment, which contradicts C2 for this learner. |
| $\eta \gt 1$ with its interval above 1 | Distractors interfere, so every unit of misalignment costs more than its share of data. |
| $\eta \lt 1$ with its interval below 1 | Distractors transfer: the shared regular-language structure helps the target. |
| G6 `SUPPORTED` | Misalignment leaves a floor that more data does not remove. That is a different law from the effective-data form and from Varun's linear term. |
| G7 `SUPPORTED` | Visitation tails predict exponents, which is direct evidence for the quanta mechanism in DFAs. |
| Shuffled data order or initialization dominates (wide seed intervals) | The grid is underpowered. Report widths and do not reinterpret thresholds. |

## 7. Compute estimate and hardware

The target is a CPU-only (central processing unit) Linux box with 4 cores, 15 GB RAM, no GPU (graphics processing unit), 4 worker processes and 1 torch thread each. Measured on 10-18-2026 on a comparable box (Intel Xeon 2.20 GHz, 2 physical cores with 4 hardware threads, 15 GB, torch 2.14.1+cpu), with 4 concurrent processes:

- **Per step:** about 70 ms for one training step of 64 strings of 17 tokens (43 ms when running alone).
- **Per evaluation:** about 2.0 s for 4,096 test strings.
- **Per run:** 1,024 steps plus 8 evaluations plus saving, about 91 s.
- **Full grid:** 512 runs / 4 workers × 91 s ≈ **3.2 h**. On 4 physical cores it should be about 2.0–2.3 h.
- **Other costs:** the pilot takes about 11 min. The optional LSTM arm (30 ms per step alone) takes about 0.5 h. Task generation and Hopcroft checks take under 1 min.
- **Wall cap:** `max_wall_seconds = 18000` (5 h). At the cap the run stops cleanly. Finished cells are atomic and are never retrained. A `--resume` into the same output directory is allowed only once, and is logged in `CKPT`.
- **Memory and disk.** Under 0.5 GB RAM per worker. Checkpoints take 4,096 × 104,002 × 4 bytes ≈ 1.70 GB (float32 `.npz`). Predictions take about 67 MB and data about 12 MB (bit-packed strings). The total is about 1.8 GB, under 2 GB. The optional LSTM arm saves only its $T = 2^{16}$ checkpoints (43 MB).

## 8. Artifacts and verification

Under `experiments/11_dfa_exact_complexity_vs_alignment/expt_v1/results/`, following the shared output contract that Experiments 14 and 15 read:

- `tasks.json`: every task with `task_id`, `family: "dfa"`, `role` (`target` or `distractor`), `K` ($n$), `K_bits`, `definition` (`n_states`, `alphabet`, transition table, accept set, `start: 0`), `acceptance_rate`, `acceptance_by_length`, `sensitivity`, `sensitivity_quartile`, `visit_freq`, `visit_tail_exponent` ($b$), and `task_token`.
- `data/<task_id>_{train,test}.npz`: pools and the shared test set as integer arrays (`bits` uint32 with bit $i$ = character $i$, $a = 0$; `length` uint8; `label` uint8).
- `mixes.json`: for each (target, $\alpha$, seed), the per-task counts at every checkpoint and the data-order seed.
- `metrics.csv` (4,096 rows), `predictions/*.npz`, `tstar.csv`, `fits.json`, `forecasts_frozen.json`, `analysis.json`.
- `checkpoints/<seed>_<target_id>_<alpha>/T_<T>.npz`: float32 weights as plain numpy arrays, so replay needs no pickle. These stay on the run box (about 1.7 GB, kept out of git by an experiment-local `.gitignore`). Their SHA-256 hashes go in `ARTIFACT_SHA256.json`. Training is bitwise deterministic for a given seed, torch version and one thread, which a unit test checks, so any checkpoint can be regenerated and checked against its hash.
- `manifest.json`: source and config SHA-256, library versions, host, wall time and the pilot record.

**`verify.py`** must report PASS before any result is written up. It:

- re-hashes the sources;
- regenerates every task from the task seed and recomputes $n$ with both Hopcroft and Moore;
- recomputes every covariate;
- regenerates pools, the test set and schedules and compares them byte for byte;
- reloads every checkpoint into a fresh model and recomputes every `metrics.csv` loss, with `max_abs_diff < 1e-9` bits;
- recomputes $T^\star$;
- refits all laws with both fitters (agreement within $10^{-4}$ relative);
- recomputes every gate from scratch.

## 9. Scope and limits

- $K$ is exact only for the DFA description language $\mathcal L_{\text{DFA}}$ and the random-DFA sampler stated here. It is not universal Kolmogorov complexity, and a different encoding would rescale $K_{\text{bits}}$ (both units are reported).
- There is one architecture, one model size, one learning rate, one input distribution and string lengths of 8–24. A different learner could rank the same automata differently, as the sensitivity and expressivity literature predicts.
- With 4 targets per level, seed-only intervals are conditional on these 16 automata. G1 adds a two-stage interval for that reason.
- The $T$ axis is examples seen along a constant-rate trajectory, capped at $2^{16}$ by the CPU budget. Behaviour past $2^{16}$ is extrapolation, and the pilot's $2^{17}$ point is disclosed but not gated.
- $\alpha$ is set by construction, and the distractors are unrelated regular languages with their own task tokens. This is the cleanest version of alignment. It is not the measured, graded alignment of real data, which Experiments 14 and 15 address.
- A positive result says how DFA complexity and target share trade off for this learner. It does not show that benchmark difficulty in large language models is mostly alignment.

**TLDR-end:** [tdc: Expt 11 design] Sixteen target regular languages with exact minimal-DFA sizes 4–32 (Hopcroft-verified, sensitivity-stratified) are each trained at target shares 0.05–1.0 against a fixed 16-DFA distractor pool, with 8 seeds and 512 CPU runs of a 104,002-parameter transformer, to test on held-out cells whether complexity raises the data bill (C1), whether alignment still dominates in units of data (C2), and whether complexity moves the scaling coefficient or the exponent (C3).
