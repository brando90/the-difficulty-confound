# Complexity versus alignment: Experiments 11–16 (design and run order)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/ideas/complexity_vs_alignment/README.md>

Status: designs and coding-agent prompts written 10-18-2026. Nothing has run yet. Each experiment's `PROTOCOL.md` is a draft until the running agent freezes it in its own commit, before any training, as in Experiments 01–09. They are numbered 11–16 because Experiment 10 is the autoregressive transformer replication of Experiment 07.

## The idea in three sentences

Brando's hypothesis (10-18-2026): Kolmogorov complexity is real, in the sense that a more complex task costs more data and compute to learn, but once a model has trained on data aligned with the target, alignment dominates how well it does. So "difficulty" is not fake; it is a learning cost that alignment can pay down. Experiments 11–16 test this with toy tasks whose complexity we can compute exactly, with alignment set by hand, and then carry the result back to addition and, through Task2Vec, toward real data.

## Why this needs new experiments

Experiments 01–09 held the task fixed and changed exposure. They showed exposure can reverse difficulty rankings, but they could not say how much of difficulty is complexity, because addition's complexity never changed: carry-chain length is a structural demand, not Kolmogorov complexity ([COMPLEXITY.md](../../archive/01_fixed_width_addition_exposure/COMPLEXITY.md), [predictability bridge](../predictability_bridge/README.md#where-kolmogorov-complexity-belongs)). To separate the two we need tasks where complexity $K$ is known exactly and varied on purpose, crossed with alignment $\alpha$ varied on purpose.

**What "exact K" means here.** Universal Kolmogorov complexity is uncomputable. We use $K_{\mathcal L}$, the exact length of the shortest description of the task in a fixed, small description language $\mathcal L$: minimal deterministic finite automaton (DFA) size, found exactly with Hopcroft's algorithm (the Myhill–Nerode theorem; Sipser, *Introduction to the Theory of Computation*, ch. 1), or the shortest program in a tiny domain-specific language (DSL), found by exhaustive search. $K_{\mathcal L}$ is computable and exact for its language. Once it is written as an encoding length in bits ($K_{\text{bits}}$, e.g. a DFA's transition table rather than its state count), an interpreter for $\mathcal L$ gives $K(x) \le K_{\text{bits}}(x) + c_{\mathcal L}$ for universal $K$; the bound can be very loose. It is not universal $K$, and every result is stated for its language. The Coding Theorem Method (CTM) tables give an independent cross-check on short strings (Experiment 12).

## The formal frame: complexity and alignment as two exchange rates on data

Let $\varepsilon = L_{\text{target}} - L^\star$ be the held-out excess loss on the target task, in bits, where $L^\star$ is the Bayes floor (0 for deterministic tasks; computed exactly for the hidden Markov model (HMM) tasks). $T$ is training examples (or tokens) seen, $\alpha \in (0, 1]$ the share of training data drawn from the target task (the rest comes from other tasks), and $K$ the target's exact complexity. All experiments fit the same nested family:

| Name | Form | Reading |
|---|---|---|
| `M0_scale` | $\varepsilon = A\,T^{-\gamma}$ | neither $K$ nor $\alpha$ matters |
| `M_align_eff` | $\varepsilon = A\,(\alpha^{\eta} T)^{-\gamma}$ | alignment acts as effective data; $\eta = 1$ is pure "only target data counts" |
| `M_align_add` | $L = a + c\,T^{-\gamma} - d\,\alpha$ | the first-order alignment law of the [alignment-scaling paper with Varun](https://github.com/madanva/alignment-scaling-laws) (no $N$ term when $N$ is fixed) |
| `M_K_coef` | $\varepsilon = A\,K^{\kappa}\,(\alpha^{\eta} T)^{-\gamma}$ | complexity moves the **coefficient** (minimum description length, MDL, picture) |
| `M_K_exp` | $\varepsilon = A\,(\alpha^{\eta} T)^{-(\gamma_0 + \gamma_1 \log_2 K)}$ | complexity moves the **exponent** (quanta picture) |
| `M_K_both` | $\varepsilon = A\,K^{\kappa}\,(\alpha^{\eta} T)^{-(\gamma_0 + \gamma_1 \log_2 K)}$ | both |

Where $N$ (model size) varies (Experiments 13-GPU and 16), every form gains Varun's $b\,N^{-\beta}$ term, so the full law is $L = a + bN^{-\beta} + c(K)\,T^{-\gamma(K)} - d\alpha$ against its effective-data counterpart.

**Two hypotheses for how complexity makes a scaling law** (Brando: "I wonder how complexity would create a scaling law"):

1. **Coefficient, from minimum description length (MDL).** For a model class with $k$ free parameters, the Bayes-mixture code spends about $\frac{k}{2}\log T$ extra bits over $T$ samples, so the per-sample excess is about $\frac{k}{2T}$ (Rissanen 1986; Barron, Rissanen and Yu 1998; Grünwald 2007, ch. 8). For a finite class of $2^{K}$ deterministic hypotheses, the Occam bound gives error $\lesssim K\ln 2 / T$. Either way $K$ multiplies the coefficient and leaves the exponent alone: $\kappa \approx 1$, $\gamma_1 = 0$.
2. **Exponent, from quanta.** If a task decomposes into sub-skills used with power-law frequencies, the learning-curve exponent is set by that frequency tail ([Hutter 2021](https://arxiv.org/abs/2102.04074); [Michaud et al. 2023](https://arxiv.org/abs/2303.13506)). A bigger automaton adds rarely visited states, which lengthens the tail, so $K$ lowers $\gamma$: $\gamma_1 \lt 0$. The mechanism is checkable: we log each target's state-visitation (or sub-program) frequency tail and test whether it predicts the fitted $\gamma(K)$.

**What these predict, in units of data.** The data needed to reach a target excess loss $\varepsilon^\star$ is $T^\star(K,\alpha) = \alpha^{-\eta}\,(A K^{\kappa} / \varepsilon^\star)^{1/\gamma}$. Alignment multiplies the data bill by $\alpha^{-\eta}$ at every $\varepsilon^\star$. Under the coefficient picture, complexity multiplies it by a constant $(K_2/K_1)^{\kappa/\gamma}$. Under the exponent picture the complexity multiplier grows without bound as $\varepsilon^\star \to 0$, so at high enough accuracy complexity eventually dominates. That makes Brando's claim precise and falsifiable:

- **C1, complexity is real:** the complexity data multiplier is $\gt 1$ with a 95% lower bound $\gt 1$.
- **C2, alignment dominates once trained:** across the pre-registered grid, the alignment multiplier range exceeds the complexity multiplier range at the pre-registered $\varepsilon^\star$, and at $\alpha = 1$ the complexity gap in loss shrinks as $T$ grows.
- **C3, which law:** `M_K_coef` versus `M_K_exp` on held-out cells. If the exponent picture wins, C2 holds only above some accuracy and the report must say where it flips.

## Shared rules for Experiments 11–16

These follow the [collaborator onboarding](../COLLABORATION_ONBOARDING.md) workflow.

- **Freeze first.** Commit `PROTOCOL.md` (with any pilot disclosed, as Experiment 09 did) before the first gated run. A one-seed calibration pilot that sets the $T$ range is allowed and must be disclosed; its seed is never reused.
- **Fit and held-out cells.** Each experiment withholds whole cells: the middle $K$ level, one middle $\alpha$ level, and the largest $T$ (extrapolation). Laws are fit per seed on fit cells and scored on held-out cells; forecasts are frozen before held-out outcomes are read.
- **Uncertainty.** 95% paired bootstrap intervals over seed blocks, 20,000 resamples, never treating items or checkpoints from one seed as independent.
- **Gate vocabulary.** Every gate ends `SUPPORTED` or `NOT_ESTABLISHED`, reported with equal prominence. The model-comparison default is the repo's: relative held-out root-mean-square error (RMSE) reduction $\ge 10\%$ with a 95% lower bound $\gt 0\%$, plus no invalid forecasts (nonfinite or below the floor).
- **Fitting.** Nonlinear least squares on $\log \varepsilon$, with a 64-start grid over exponents, bounds $\gamma \in (0, 3]$, $\eta \in [0, 3]$, and an independent second fitter (`independent_fit.py`) that must agree to $10^{-4}$ relative before results are reported.
- **Verify.** `verify.py` reloads saved checkpoints, recomputes held-out losses to `max_abs_diff < 1e-9` bits, recomputes every exact $K$ from scratch, and refits the laws.
- **Retain everything.** No seed, task or forecast is dropped. Outputs: `RESULTS_<slug>.md`, `CKPT_<slug>.md` (dated log), `ARTIFACT_SHA256.json`, a figure, then archive under `experiments/archive/` as before.

## The six experiments

| Expt | Toy and folder | Complexity $K$ | Alignment $\alpha$ | Hardware (CPU = central processing unit, GPU = graphics processing unit) | Who runs it |
|---|---|---|---|---|---|
| 11 | Regular languages: [`11_dfa_exact_complexity_vs_alignment`](../../11_dfa_exact_complexity_vs_alignment/) | minimal DFA states (exact, Hopcroft) | target-DFA share of the training mix | CPU, about 2–4 h on 4 cores | Brando's assistant |
| 12 | Bit-string functions: [`12_dsl_shortest_program_complexity_vs_alignment`](../../12_dsl_shortest_program_complexity_vs_alignment/) | shortest program in a 12-primitive DSL (exact, exhaustive search), CTM cross-check | target-function share | CPU, about 2–4 h | Brando's assistant |
| 13 | Hidden Markov model mixtures in the style of GINC (Generative In-Context learning dataset): [`13_ginc_hmm_complexity_vs_alignment`](../../13_ginc_hmm_complexity_vs_alignment/) | target-HMM description length; exact entropy-rate floor | target-concept share | small CPU version; GPU-scale version | assistant (CPU); Laolu (GPU) |
| 14 | Task2Vec bridge: [`14_task2vec_vs_exact_complexity`](../../14_task2vec_vs_exact_complexity/) | reuses 11 and 12 | reuses 11 and 12 | CPU, about 1–2 h | Brando's assistant |
| 15 | Recovering $\alpha$: [`15_alignment_alpha_recovery`](../../15_alignment_alpha_recovery/) | reuses 11 and 12 | known true $\alpha$ versus measured $\hat\alpha$ | CPU, about 2 h (scoring plus a 160-run graded-relevance grid) | Brando's assistant |
| 16 | Addition, crossed: [`16_carry_alignment_crossed_arithmetic`](../../16_carry_alignment_crossed_arithmetic/) | carry-chain length (structural proxy, **not** $K$) | target-carry-type share | CPU pilot; GPU transformer arm | Laolu (SNAP or Sherlock) |

## Run order

1. **11** (DFA). The cleanest test of C1–C3; everything else builds on its pipeline.
2. **12** (DSL programs). Same logic with a second, independent exact-$K$ language, so a result is not an artifact of automata.
3. **14** and **15**, in parallel. Both are mostly analysis of the checkpoints and data from 11 and 12.
4. **13-small** (HMM, CPU). Brings in the GINC setting from Brando's diversity work and an exact entropy floor.
5. **16** and **13-GPU** on SNAP (Stanford Network Analysis Project) GPU nodes (skampere1/2 or mercury1/2 by direct secure shell, SSH) or Sherlock (`gpu` partition). Launch scripts are in each folder's `launch/`. These add model size $N$, so they test the full $L = a + bN^{-\beta} + c(K)T^{-\gamma(K)} - d\alpha$ law.

Each experiment's `PROMPT.md` is a self-contained prompt for a coding agent (Claude Code or Codex) to run that experiment end to end: build, test, freeze, run, verify, report, archive.

## How this connects to Task2Vec and real data

Brando: "task2vec is basically a way for us to make the toy experiments eventually move to real data experiments." On real data neither $K$ nor $\alpha$ is known. Experiment 14 checks, on toys where both are known, whether the Task2Vec embedding norm tracks $K$ and whether Task2Vec distance between a training mix and the target tracks $\alpha$; Experiment 15 does the same for the alignment-scaling paper's own $\alpha$ measure. Only measures that pass there should be used as $\hat K$ and $\hat\alpha$ on real fine-tuning data. What Task2Vec does and does not say about Kolmogorov complexity is in [TASK2VEC_KOLMOGOROV.md](TASK2VEC_KOLMOGOROV.md).

## Scope

A positive result says how complexity in a stated description language and hand-set alignment trade off for these learners and budgets. It does not show that benchmark difficulty in large language models is mostly alignment, and it does not estimate universal Kolmogorov complexity.

**TLDR-end:** [tdc: Expts 11–16 designs] Six pre-registered experiments cross exactly computable task complexity with hand-set data alignment to test whether complexity raises the data bill while alignment dominates final performance, and whether complexity enters scaling laws through the coefficient (MDL) or the exponent (quanta); 11, 12, 14, 15 and small 13 run on CPU, 16 and large 13 on SNAP or Sherlock GPUs.
