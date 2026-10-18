# The Difficulty Confound

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/README.md>

**TLDR:** How much of apparent task difficulty comes from training exposure? We test this with controlled interventions on training data while holding the task and test items fixed. Exposure comparisons match architecture and budget; a second study varies model size to test held-out forecasts.

**Is a task hard—or has a model simply not learned it yet?**

This project investigates the hypothesis that much of what we call task difficulty in AI evaluation is explained by training data: its coverage, frequency, and relevance to the evaluated task.

Difficulty may be real, but observed failure alone does not establish that a task is intrinsically difficult. It may also reflect unfamiliarity, missing prerequisites, or a mismatch between training and evaluation distributions.

## Research question

How much does measured task difficulty change with controlled training exposure, while the skill being evaluated stays fixed?

## Core hypothesis

Targeted training can substantially reduce apparent difficulty, including on held-out examples. If so, benchmark difficulty should be interpreted relative to a model's training history and evaluation conditions.

This is a hypothesis to test, not an assumed conclusion.

## First experiment

[experiments/archive/01_fixed_width_addition_exposure/](experiments/archive/01_fixed_width_addition_exposure/) trains small neural networks from scratch to add two three-digit numbers. Every condition answers exactly the same held-out questions. Only the training distribution over carry-chain lengths changes.

- Same 29,480-parameter network, initial parameters within each seed, 16,000 distinct training pairs, 3,000 updates, and 256 examples per update.
- Eight paired seeds and four conditions: scarce long carries, balanced coverage, abundant long carries, and shuffled training labels.
- Four thousand fixed test pairs, with swapped operands excluded from training too.
- Exact integer answers, with an independent addition implementation checked exhaustively.
- A prospective criterion for whether exposure recovers most of the initial accuracy gap, without materially hurting no-carry performance.

**Result:** Length-three-carry accuracy rose from 9.60% to 94.89% with balanced training, recovering 94.34% [92.74, 95.86] of the predefined gap across eight seeds. The stricter criterion did not pass: no-carry accuracy fell 6.08 points, exceeding the allowed five-point decline. [Full results](experiments/archive/01_fixed_width_addition_exposure/RESULTS_addition_exposure.md).

**Status:** all 32 runs completed and verified. [Live results](experiments/archive/01_fixed_width_addition_exposure/results.md) · [Locked protocol](experiments/archive/01_fixed_width_addition_exposure/PROTOCOL.md).

## Held-out predictive-law experiment

[experiments/archive/02_heldout_exposure_difficulty_laws/](experiments/archive/02_heldout_exposure_difficulty_laws/) completed 200 training trajectories across eight seeds, five network widths, and five exposure mixtures. The same exact addition task and questions are retained; two widths and two mixtures are withheld from law fitting. The baseline adapts the additive form from the alignment-scaling project with Varun.

**Primary result:** adding both log exposure and independently probed starting difficulty reduced joint held-out root mean squared error by **6.80% [-3.52, 14.80]**, missing the prospective 10% and positive-lower-bound requirements. The predeclared difficulty-only ablation improved **15.80% [10.27, 20.52]**; this secondary result needs fresh replication. Intervals are 95% paired bootstrap intervals over eight seed blocks, with 384 joint aggregate targets per law; p-val=n/a. The difficulty-only law still made 52/384 invalid negative-loss forecasts. [Full results and all ablations](experiments/archive/02_heldout_exposure_difficulty_laws/RESULTS_heldout_laws.md).

On 24 actual mathematics observations from Varun's alignment-law repository, a training-breadth proxy worsened joint held-out error by 1.14% (relative reduction **-1.14% [-9.58, 0.57]**, descriptive 95% interval over four size blocks; p-val=n/a). Independently measured starting difficulty was unavailable there. [Companion report, authorized observations, and predictions](experiments/archive/02_heldout_exposure_difficulty_laws/RESULTS_actual_math.md).

## Answer-matched ranking-reversal experiment

[experiments/archive/03_answer_matched_alignment/](experiments/archive/03_answer_matched_alignment/) completed 200 trajectories with eight fresh seeds, five widths and five exposure mixtures. Every intervention has 4,680 pairs with exactly the same answer histogram, and both evaluation carry strata have 936 questions with matching answer histograms.

**Primary result:** the difficulty ranking reverses at every width. No-carry questions lead by **25.30 [23.14, 27.26] percentage points** when they receive 90% of intervention exposure; two-carry questions lead by **24.58 [22.88, 26.19] points** when they receive 90%. Both pass the frozen 10-point and uncertainty criteria. The alignment contrast is **24.94 [23.27, 26.40] points**, versus a residual carry main effect of **0.36 [-0.60, 1.40] points**, for this particular 9:1 intervention and accuracy scale. Intervals resample eight paired seed blocks, averaging widths within seed, conditional on the fixed test set; p-val=n/a.

**Predictive counterevidence:** adding independently probed starting difficulty to the positive log-linear law raises joint held-out root mean squared error from **0.4727 [0.3212, 0.5990] to 292.7941 [0.3588, 507.0797] bits/answer**, over 192 targets per law. Relative error reduction is **-61835.64% [-140226.76, 27.71]**; the frozen predictive gate fails. One seed dominates the extrapolation failure; no seed or forecast was removed. Merely adding exposure also worsens joint prediction by **9.58% [7.86, 14.25]**. All intervals use the same eight paired seed blocks; p-val=n/a.

[Complete results, plot and failure diagnosis](experiments/archive/03_answer_matched_alignment/RESULTS_matched_alignment.md). Training changes observed difficulty without changing the addition task, evaluation items or their formal description complexity. Answer matching still leaves substantial input-digit differences; this establishes a training-distribution effect in controlled addition, not that complexity is irrelevant or that data dominates across language models.

## Extrapolation-repair and feature-value experiment

[experiments/archive/04_monotone_ridge_heldout_addition/](experiments/archive/04_monotone_ridge_heldout_addition/) completed 200 fresh trajectories and 40 shared warm starts across eight new seeds (`3101`–`3808`), unseen evaluation splits disjoint from Experiment 03, held-out mixtures `.2/.8`, and held-out widths `96/384`. It freezes a constrained monotone ridge law (`stable_difficulty`, with nonpositive slopes on log parameters, updates, and exposure, and starting-difficulty elasticity in `[0,1]`) and separates the extrapolation-repair gate (`stable_difficulty` vs `legacy_difficulty`) from the incremental feature-value gate (`stable_difficulty` vs `stable_base`).

**Extrapolation repair:** **SUPPORTED**. `stable_difficulty` lowers joint held-out root mean squared error from **1,935,297,769,746.72 [619.43, 3,352,033,360,529.69] to 0.4835 [0.3760, 0.5780] bits/answer** (**100.00% [99.90, 100.00]** reduction; 0/192 invalid predictions), improving at both width 96 (`3.2963 → 0.6801`) and width 384 (`2.737e12 → 0.0718`).

**Feature value under shared warmup:** **NOT_ESTABLISHED**. Relative to `stable_base` (**0.5010 [0.3894, 0.5972]** bits/answer), `stable_difficulty` reduces joint RMSE by **3.48% [1.12, 6.24]** (improving at both held-out widths: 3.45% at width 96 and 6.04% at width 384), which is statistically positive but below the predeclared 10% practical threshold because all warm starts share the same balanced (`q=0.50`) warmup history. The secondary answer-matched ranking reversal is replicated (**SUPPORTED**: **24.07 [22.80, 25.46]** and **24.48 [22.77, 26.06]** percentage points). [Complete results and diagnostics](experiments/archive/04_monotone_ridge_heldout_addition/RESULTS_stable.md).

## Heterogeneous pretraining-history and no-harm retention experiment

[experiments/archive/05_pretraining_history_and_retention/](experiments/archive/05_pretraining_history_and_retention/) tests why starting difficulty added only 3.48% in Experiment 04 and whether targeted post-training can close a pretraining-induced carry gap without hurting no-carry retention. Across eight fresh seeds (`4101`–`4808`), five widths (fit `32/64/128`, held-out `96/384`), and five pretraining histories ($q_{\text{pre}} \in \{0.1, 0.5, 0.9\}$ fit; $\{0.2, 0.8\}$ held-out), 200 checkpoints are pretrained for 1,500 updates, probed on an independent 936-item set, and then post-trained for 1,000 updates on balanced coverage ($q_{\text{post}}=0.50$).

**Incremental feature value under heterogeneous history:** **SUPPORTED**. Adding probed starting difficulty (`stable_difficulty` vs `stable_base`) reduces joint held-out post-training RMSE from **0.1902 [0.1425, 0.2323] to 0.1004 [0.0688, 0.1268] bits/answer** across 192 held-out targets—a **47.20% [42.88, 53.08] reduction** with 0/192 invalid predictions, improving at both held-out widths (`0.2690 → 0.1420` at width 96; `0.0061 → 0.0046` at width 384) and even outperforming an oracle law given true pretraining mixture fractions (`stable_oracle_history`, **0.1580 [0.1083, 0.1991]** bits/answer) by **36.43% [32.50, 39.58]**.

**Targeted remediation and strict no-harm retention:** **SUPPORTED**. On checkpoints pretrained with scarce two-carry exposure ($q_{\text{pre}}=0.10$), initial accuracy is **95.55% [94.36, 96.60]** on no-carry vs **56.54% [54.68, 58.65]** on two-carry (initial gap **39.01 [36.31, 41.57]** percentage points). Balanced post-training raises two-carry accuracy to **90.70% [88.79, 92.43]** (**+34.16 [31.59, 36.68]** points, recovering **87.55% [83.60, 90.72]** of the initial gap) while control no-carry accuracy changes by **+1.09 [+0.33, +1.89] percentage points** (`95.55% → 96.64%`), passing the strict $\ge -5.00$-point retention gate at every width. [Complete results and plot](experiments/archive/05_pretraining_history_and_retention/RESULTS_history_retention.md).

## Compositional carry-marginal matched experiment

[experiments/archive/06_compositional_carry_marginal_matched/](experiments/archive/06_compositional_carry_marginal_matched/) tests whether the difficulty ranking reversal in Experiments 03–05 was driven by per-column carry frequencies or input-digit marginal shifts rather than compositional exposure. Across eight fresh seeds (`6101`–`6808`), five widths, and five mixtures, it contrasts an **Isolated Single-Carry Regime ($S_{\text{iso}}$: 50% `(1,0)` + 50% `(0,1)`)** against a **Paired Zero/Double-Carry Regime ($S_{\text{comp}}$: 50% `(0,0)` + 50% `(1,1)`)** across 180 exact-matched answers (7,200 training pairs/mixture). Every mixture $q \in \{0.1, 0.2, 0.5, 0.8, 0.9\}$ holds exact per-column carry rates fixed at **50.00% ones (`3600/7200`), 50.00% tens (`3600/7200`), and 0.00% hundreds**, cutting input-digit Total Variation (TV) distance by more than $11\times$ (from mean `0.443` in Expts 03–04 to mean `0.0398`, range `0.0049`–`0.0972`).

**Pure compositional ranking reversal (`(1,0)+(0,1)` vs `(1,1)`):** **SUPPORTED**. When training favors isolated single carries ($q=0.10$), single-carry accuracy exceeds cascading two-carry accuracy by **22.35 [20.60, 24.00] percentage points** (`93.70%` vs `71.35%`); when training favors paired two-carry chains ($q=0.90$), cascading two-carry accuracy exceeds single-carry accuracy by **17.74 [14.62, 21.12] percentage points** (`92.13%` vs `74.39%`), reversing at all five widths. Full regime accuracy ($S_{\text{iso}}$ vs $S_{\text{comp}}$) also reverses (**SUPPORTED**: **22.39 [20.78, 23.93]** and **18.07 [14.78, 21.61]** points). Because all trajectories in Experiment 06 again share a balanced (`q=0.50`) warmup, starting difficulty reduces joint RMSE by only **1.94% [-3.10, 6.50]** (**NOT_ESTABLISHED**), confirming the contrast with Experiment 05. [Complete results and plot](experiments/archive/06_compositional_carry_marginal_matched/RESULTS_compositional.md).

## Prequential MDL and heterogeneous carry-marginal matched experiment

[experiments/archive/07_prequential_mdl_and_length_transfer/](experiments/archive/07_prequential_mdl_and_length_transfer/) unifies Experiment 05's heterogeneous pretraining histories ($q_{\text{pre}} \in \{0.1, 0.2, 0.5, 0.8, 0.9\}$) with Experiment 06's $50\%/50\%$ per-column carry-marginal matched design (input-digit TV `0.0398`), while recording exact **prequential (online) MDL code length** ([Voita and Titov, 2020](https://aclanthology.org/2020.emnlp-main.14/)) along the 1,000-step balanced adaptation trajectory across eight fresh seeds (`7101`–`7808`) and five widths.

**Conditional loss and prequential MDL predictive laws:** **SUPPORTED**. Adding probed starting difficulty (`stable_difficulty` vs `stable_base`) reduces joint held-out conditional loss RMSE by **36.22% [32.45, 40.51]** (`0.1470 → 0.0938` bits/answer, vs `1.94%` under shared warmup in Expt 06), outperforming `stable_oracle_history` (`0.1191`) by **21.26% [16.50, 25.17]**, and reduces joint held-out prequential online MDL RMSE by **56.93% [55.01, 58.67]** (`0.1823 → 0.0785` bits/item).

**Prequential online MDL reversal and pure compositional remediation:** **SUPPORTED** (Gates 2a and 3) / **NOT_ESTABLISHED** (Gate 4). Online adaptation code length reverses across all five widths between $q_{\text{pre}}=0.10$ (`0.4423 [0.4043, 0.4824]` bits/item) and $q_{\text{pre}}=0.90$ (`0.1773 [0.1525, 0.2016]` bits/item), and pre-intervention accuracy reverses by **46.06 [44.04, 48.26]** and **42.05 [39.95, 43.83] percentage points**. Balanced post-training recovers **104.29% [101.45, 107.53]** of the initial gap while single-carry control accuracy rises by **+5.13 [+4.13, +6.38] percentage points** (`79.11% → 84.24%`), though pooled 5-width target accuracy reaches **81.09% [79.51, 82.88]** (`96.91%–99.43%` at widths $\ge 96$, `27.80%` at bottleneck width 32), missing the frozen pooled 85% target line, so the remediation gate as a whole is `NOT_ESTABLISHED`. [Complete results and plot](experiments/archive/07_prequential_mdl_and_length_transfer/RESULTS_mdl_history.md).

## Reinforcement learning with verifiable reward (RLVR) utility-law experiment

[experiments/archive/08_rlvr_verifiable_reward_utility_laws/](experiments/archive/08_rlvr_verifiable_reward_utility_laws/) replaces supervised post-training with **Reinforcement Learning with Verifiable 0/1 Exact-Sum Rewards (RLVR / leave-one-out GRPO, group size $G=16$ rollouts per prompt)** across 200 pretrained checkpoints (`8101`–`8808`), directly testing the RL environment utility bridge (`rl-4-code-verification`).

**Post-RLVR loss forecast and RL improvement utility reversal:** **SUPPORTED**. Adding probed starting difficulty reduces joint held-out post-RLVR conditional loss RMSE by **39.87% [36.20, 50.17]** (`0.7809 → 0.4696` bits/answer, 0 invalid forecasts). Furthermore, realized RL improvement utility $U = \text{Acc}_T - \text{Acc}_0$ reverses across pretraining histories at all five widths: under $q_{\text{pre}}=0.10$, RLVR yields **+29.07 [26.32, 32.33] percentage points** more gain on $S_{\text{comp}}$ (`+35.52 pp`) than on $S_{\text{iso}}$ (`+6.45 pp`); under $q_{\text{pre}}=0.90$, RLVR yields **+32.70 [30.18, 35.20] percentage points** more gain on $S_{\text{iso}}$ (`+39.16 pp`) than on $S_{\text{comp}}$ (`+6.46 pp`).

**Pure 0/1 RLVR remediation and natural no-harm retention:** **SUPPORTED**. Pure 0/1 reward optimization recovers **78.23% [72.58, 84.23]** of the initial pure compositional gap (`+35.49 pp` on cascading two-carry `(1,1)`), while single-carry control accuracy *improves* by **+6.45 [+4.69, +7.91] percentage points** (`77.81% → 84.26%`), because group-relative RL assigns zero advantage ($\hat{A}_{b,g}=0$) whenever all $G=16$ rollouts on an already-mastered item succeed.

**Additive RL utility law vs ceiling saturation:** **NOT_ESTABLISHED** pooled (**-15.78% [-35.91, 3.15]**). At interpolated held-out width 96, adding $\log \rho_{m_0}$, headroom $H_{m_0}$, and $\log D_{m_0}$ reduces RL utility RMSE by **48.56% [41.65, 57.30]** (`14.59 → 7.50` pp, beating oracle history `8.53` pp by **12.01%**). However, at extrapolated near-ceiling width 384 (where $\text{Acc}_0$ is already $78\%\text{–}99.8\%$), the *additive linear* utility formula from `rl-4-code-verification` overshoots remaining headroom $1 - \text{Acc}_0$ (`19.71 → 27.38` pp), diagnosing the exact need for a headroom-bounded RL utility law. [Complete results and plot](experiments/archive/08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md).

## Headroom-bounded RL utility-law experiment

[experiments/archive/09_headroom_bounded_rlvr_utility_law/](experiments/archive/09_headroom_bounded_rlvr_utility_law/) replaces Experiment 08's additive RL utility law with a **headroom-bounded fractional-recovery law**, $`\widehat U = (1-\text{Acc}_0)(1-e^{-r_\theta})`$, which cannot forecast more improvement than the remaining headroom. It also RL-trains each of 200 fresh starting checkpoints (`9101`–`9808`) at four rollout group sizes $G \in \{4, 8, 16, 64\}$ under a fixed budget of 4,096 rollouts per update (800 trajectories).

**Structural repair:** **SUPPORTED**. Bounding by headroom reduces joint held-out RL utility RMSE by **73.72% [70.43, 77.26]** (`17.62 → 4.63` pp; 128 → 0 impossible forecasts). At near-ceiling width 384, error falls from `20.63` to `0.41` pp, the binomial noise level of the observed outcome.

**Probe features and signal density:** **NOT_ESTABLISHED**. Inside the bound, starting-checkpoint probe features improve forecasts by **8.10% [2.52, 14.39]**, below the 10% gate (an exploratory refit on Experiment 08 had suggested 16.53%). Larger groups raised probe signal density but lowered realized utility at fixed budget (`19.08` pp at $G=4$ vs `14.89` pp at $G=64$), and knowing $\rho(G)$ adds only **0.96% [0.44, 1.54]** beyond knowing $G$. Experiment 08's utility reversal (`29.86` / `34.49` pp) and RLVR remediation (**77.66%** recovery, **+5.92 pp** control) replicate. [Complete results](experiments/archive/09_headroom_bounded_rlvr_utility_law/RESULTS_headroom_bounded.md).

## Proposed approach

1. **Define a narrow skill.** Specify what the task is intended to measure and which additional abilities it requires.
2. **Construct task families.** Vary proposed difficulty factors while keeping the target skill and evaluation criteria consistent.
3. **Manipulate training exposure.** Vary the amount, diversity, and relevance of training examples under controlled conditions.
4. **Evaluate generalization.** Test on held-out examples and distinguish gains on familiar templates from transfer to new structures.
5. **Measure the remaining difficulty.** Identify which performance gaps shrink with exposure and which persist.

Holding a skill fixed is itself an empirical challenge. We will examine whether harder variants introduce new prerequisites or change what the task measures.

## Key comparisons

- Targeted training versus unrelated training with matched training budgets.
- More examples versus more diverse examples.
- Familiar task structures versus held-out structures.
- Performance before and after training across proposed difficulty levels.
- Single-skill tasks versus tasks requiring several skills together.

## What would support the hypothesis?

- Targeted exposure substantially reduces performance gaps between proposed difficulty levels.
- Improvements transfer beyond memorized examples.
- Task difficulty rankings change with training history.
- Exposure explains a substantial share of performance variation under the tested conditions.

## What would challenge the hypothesis?

- Difficulty gaps persist despite substantial, relevant training.
- Improvements remain confined to familiar examples or templates.
- Proposed difficulty factors predict held-out performance consistently across training conditions.

## Scope and limitations

This project does not assume that all difficulty reduces to data. Reasoning demands, computational resources, task composition, and evaluation conditions may also matter.

Claims will be limited to the models, task families, and training interventions tested. Controlled experiments can establish effects of exposure in those settings; they cannot, by themselves, reveal the training histories of unrelated models.

## Status

Nine controlled experiments are complete and archived with full artifacts:
- **Experiments 03, 04, 05, 06, 07, and 08** support the prespecified difficulty-ranking reversal under exact answer matching—including **Experiments 06, 07, and 08**, which hold exact per-column carry rates fixed at 50% ones / 50% tens and reduce input-digit marginal TV distance to `0.0398`, isolating a pure compositional exposure reversal across final accuracy, prequential online MDL code length, and verifiable-reward RL utility ($U = \text{Acc}_T - \text{Acc}_0$).
- **Experiments 05, 07, and 08** demonstrate that under heterogeneous pretraining histories ($q_{\text{pre}}$), probed starting difficulty reduces joint held-out conditional loss RMSE by **47.20%** (Expt 05), **36.22%** (Expt 07), and **39.87%** (Expt 08 under RLVR)—plus **56.93%** on prequential online MDL (Expt 07)—compared with 3.48% (Expt 04) and 1.94% (Expt 06) when starting models share a constant warmup. Control accuracy on previously mastered strata rises after supervised balanced post-training (**+1.09 pp** in Expt 05, **+5.13 pp** in Expt 07) and pure 0/1 verifiable-reward RL (**+6.45 pp** in Expt 08). The full remediation gate passes in Expts 05 and 08 but is `NOT_ESTABLISHED` in Expt 07 (pooled target accuracy 81.09%, below 85%). Experiment 05's extrapolation-repair gate is also `NOT_ESTABLISHED` (-0.82% [-1.11, -0.57]); its unconstrained law did not blow up, so there was nothing to repair.
- **Experiment 09** shows that bounding the RL utility law by remaining headroom repairs its near-ceiling extrapolation failure (**73.72% [70.43, 77.26]** RMSE reduction; 128 → 0 impossible forecasts), while probe features inside the bound (**8.10% [2.52, 14.39]**) and signal density beyond group size (**0.96% [0.44, 1.54]**) do not reach the 10% gate (`NOT_ESTABLISHED`).

## What stays fixed—and what does not

The operational construct is exact fixed-width addition. Input/output lengths, notation, target function, evaluator, and test items stay fixed within each experiment. Exposure comparisons at each width and checkpoint hold architecture, optimizer, and training budget fixed. The predictive-law study deliberately varies widths and budgets to measure transfer. The learned model changes: changing what it knows is the intervention. This does not prove that latent psychological skills or strategies are identical across conditions.

A positive result would establish a causal effect of the specified training-distribution intervention in this synthetic setting. It would not establish that most difficulty across language-model benchmarks is caused by data, nor that training exposure is the only explanation for a particular proprietary model's failures.

## Complexity belongs in the argument

Kolmogorov complexity is the length of the shortest program producing an object, relative to a fixed universal machine. It is not generally computable. For an unchanged question, answer, and task specification, its formal conditional complexity stays unchanged across our training conditions, even when model accuracy changes.

We measure both **conditional predictive code length** (how many bits the trained model needs to encode the correct answer given the input) and, in [Experiment 07](experiments/archive/07_prequential_mdl_and_length_transfer/RESULTS_mdl_history.md), **prequential (online) minimum description length** ([Voita and Titov, 2020](https://aclanthology.org/2020.emnlp-main.14/)), which accumulates the online code length spent transmitting labels along the adaptation stream. [Theory and reviewer-facing limitations](experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md).

## Connection to predictive scaling laws

Difficulty can enter a predictive law as a property of the task, starting model, and training history. Our [connection to the projects with Varun](experiments/ideas/predictability_bridge/README.md) separates structural demand, known training exposure, and difficulty measured at a particular checkpoint. It specifies extensions to the existing environment-utility model, leakage controls, and held-out-scale comparisons. Across Experiments 02–08, we show that constrained monotone ridge eliminates log-linear extrapolation blow-up (Expt 04), probed starting difficulty cuts held-out post-training loss RMSE by $36.22\%\text{–}47.20\%$ and prequential MDL RMSE by $56.93\%$ when starting checkpoints have heterogeneous pretraining histories (Expts 05, 07, 08), and in pure 0/1 verifiable-reward RL (Expt 08), adding stochastic rollout signal density $\rho_{m_0}$ and starting difficulty $D_{m_0}$ cuts RL improvement utility RMSE by **48.56%** at interpolated width 96 while exposing ceiling-headroom overshoot in the additive linear utility form at near-ceiling width 384.

## Publication authorization

On 10-16-2026, Brando authorized publication of all code, experiment data, results, and supporting artifacts for this project, including the original companion inputs copied from the alignment-scaling repository: “feel free to publish everything, no need to ask me again.” Treat this as standing project authorization; do not ask again merely because these project artifacts originated in a private source repository. Credentials and unrelated personal material remain outside this research release.

## Reproduce

Use Python with the [pinned numerical dependencies](experiments/archive/01_fixed_width_addition_exposure/expt_v1/requirements.txt). Recompute and verify the published results from the repository root:

```bash
python experiments/archive/01_fixed_width_addition_exposure/expt_v1/analyze.py
python experiments/archive/01_fixed_width_addition_exposure/expt_v1/verify_artifacts.py
```

To train a fresh reproduction, choose a new output directory:

```bash
python experiments/archive/01_fixed_width_addition_exposure/expt_v1/run.py --check-only
python experiments/archive/01_fixed_width_addition_exposure/expt_v1/run.py --output /tmp/difficulty-reproduction
python experiments/archive/01_fixed_width_addition_exposure/expt_v1/analyze.py --output /tmp/difficulty-reproduction
```

The training script refuses to overwrite an existing run. Configuration, split seeds, per-item predictions, training manifests, source hashes, and runtime versions support reproduction. The analysis reads the per-item tables, independently recomputes accuracy, and rejects incomplete runs.

## Related work

Training coverage and compositional generalization already have a substantial literature. Our contribution must go beyond rediscovering that training helps: quantify an exposure intervention at matched compute, retain the same test items, and state precisely what fraction of which gap it explains.

- [Lake and Baroni (2018): Generalization without Systematicity](https://proceedings.mlr.press/v80/lake18a.html).
- [Keysers et al. (2020): Measuring Compositional Generalization](https://arxiv.org/abs/1912.09713).
- [Voita and Titov (2020): Information-Theoretic Probing with Minimum Description Length](https://aclanthology.org/2020.emnlp-main.14/).

## License

[Apache License 2.0](LICENSE).
