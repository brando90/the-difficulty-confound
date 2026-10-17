# Exposure-aware difficulty in predictive scaling laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/ideas/predictability_bridge/README.md>

**TLDR:** Connect training exposure to checkpoint-relative difficulty, then ask whether it improves forecasts beyond the existing scale, budget, signal-density and coverage coordinates. Keep structural complexity separate. Across five held-out predictive-law studies (Experiments 02–06), we show that: (1) unconstrained log-linear difficulty laws can suffer catastrophic extrapolation blow-up on larger held-out scales (Expt 03), (2) a constrained monotone ridge law (`stable_difficulty`) eliminates extrapolation blow-up with 0 invalid predictions (Expt 04), and (3) independently probed starting difficulty $D_{m_0}$ adds small incremental value ($1.94\%\text{–}3.48\%$) when all warm starts share a constant training history (Expts 04 and 06), but reduces joint held-out RMSE by **47.20% [42.88, 53.08]**—outperforming even an oracle pretraining-mixture law by **36.43% [32.50, 39.58]**—when starting checkpoints have heterogeneous prior training histories (Expt 05).

Created: 10-16-2026. Status: original feature export complete; held-out predictive-law series (Experiments 02–06) complete and archived.

## Where this connects to the work with Varun

The [Reinforcement-learning (RL) environment project](https://github.com/madanva/rl-env-scaling) names Varun and Brando in its team and already defines difficulty relative to the current model, starting checkpoint and pretraining history in [docs/axes/difficulty.md](https://github.com/madanva/rl-env-scaling/blob/main/docs/axes/difficulty.md). Its candidate measures include per-task success probability and productive uncertainty. This is the natural home for the connection. The local planning repository points to implementation in [rl-4-code-verification](https://github.com/brando90/rl-4-code-verification).

The implemented [utility model](https://github.com/brando90/rl-4-code-verification/blob/main/src/rl4cv/utility_model.py) forecasts held-out RL improvement:

$$
\widehat U=b_0+b_Nx_N+b_Cs(C)+b_\rho s(C)z(\log\rho)+b_Hs(C)z(H),
\qquad s(C)=C/(C+C_0).
$$

Here $N$ is model scale, $C$ is the generated-token budget, $\rho$ is usable signal density, and $H$ is effective coverage. The standardization $z$ uses training-fold statistics. The source defines $U$ as improvement from the starting checkpoint, rather than final accuracy. Those distinctions must survive any extension.

There is also the [alignment-scaling project](https://github.com/madanva/alignment-scaling-laws), whose paper with Varun models loss as $L=a+bN^{-\beta}+cT^{-\gamma}-d\alpha$. Here $T$ is the training-token budget and $\alpha$ measures train–target alignment. Its [rebuttal fitting code](https://github.com/madanva/alignment-scaling-laws/blob/main/experiments/00_neurips2026_rebuttal/scripts/fitlib.py) also tests $-d\alpha N^{-\delta}$. The existing rebuttal reports evidence against a scale-independent alignment benefit. Thus alignment–capacity interaction is already relevant; a new difficulty term should not assume additivity or duplicate alignment without testing incremental predictive value. These are descriptions of retrieved project artifacts, not a fresh reanalysis or endorsement of every paper claim.

## Three different variables

| Coordinate | Example | Available when? | Interpretation |
|---|---|---|---|
| Structural demand $c_i$ | Carry-chain length; fixed interpreter step count | Before model evaluation | A specified computational property, not a validated universal difficulty scale |
| Exposure $e_{i,D}$ | Training count of an item's structural stratum; similarity to the training distribution | Before evaluation, if training data are known | A property of the item–training-distribution pair |
| Empirical difficulty $d_{i,m_0}$ | Failure probability under the starting checkpoint and fixed evaluation protocol | From a separate starting-policy probe | A model-, history-, and protocol-relative measurement |

Condition labels, benchmark identity and random seed belong in grouping metadata. They are not substitutes for measured causal coordinates. Historical exposure for a proprietary model is unknown unless independently documented; a proxy must remain labelled as a proxy.

The proposed mechanism is **training history → starting-policy success probabilities → available reward signal and coverage → later learning utility**. This is a hypothesis, not a mediation result from the addition experiment. For binary independent responses, a group of size $G$ contains both successes and failures with probability $1-p^G-(1-p)^G$; the [existing experience module](https://github.com/brando90/rl-4-code-verification/blob/main/src/rl4cv/experience.py) already implements this link. One deterministic greedy answer per addition item does not estimate that item's stochastic $p$.

## Smallest useful extension

Keep the existing law as the baseline. On an independent, fixed starting-checkpoint probe, measure the distribution of success probabilities and the proposed structural and exposure coordinates. Compare, in order:

1. Existing scale/budget/$\rho,H$ model.
2. Baseline plus one exposure coordinate, $s(C)\gamma_e z(e)$.
3. Baseline plus one structural-demand coordinate, $s(C)\gamma_c z(c)$.
4. Both coordinates and a prespecified exposure × structure interaction, $s(C)\gamma_{ec}z(e)z(c)$.
5. Only if supported by enough independent scales, test a prespecified exposure × scale interaction.

These are candidate models, not fitted findings. For environment-level utility, aggregate item features with fixed target-item weights; do not let an intervention silently change the evaluation mixture. A common fixed structural distribution has a constant mean, so its main effect cannot be identified from aggregate utility alone. Retain stratum-level outcomes/interactions or introduce independently varied, prespecified environments. Coverage and signal density may already absorb exposure effects; no added forecast benefit would not refute the controlled causal exposure effect.

Use held-out scales **and** held-out exposure distributions, with whole runs/checkpoints held together. Include a held-out task family before claiming transfer. Fit preprocessing, feature selection and hyperparameters inside each training fold. Compare held-out prediction error, ranking and allocation regret against the implemented baseline. Bootstrap paired differences at the independent training-run level; report counts of seeds, scales and environments and intervals conditional on the chosen task set. Require prediction improvement and calibration, not merely a coefficient with a small p-value. Freeze the model comparison and evaluation splits before new outcomes are inspected.

**Leakage rule:** final test accuracy, final predictive code length and difficulty estimated from the target outcome cannot serve as predictors of that same outcome. A starting-policy probe can be a predictor if its budget and access are part of the prospective protocol. Report its cost; do not call it a prediction available before probing.

## What the existing addition experiment can supply now

The fixed three-digit addition experiment crossed eight seeds, four training distributions and four carry strata. Its [full report](../../archive/01_fixed_width_addition_exposure/RESULTS_addition_exposure.md) records the large exposure effect and the missed no-carry retention criterion. The fixed task and held-out questions make it a controlled test of exposure dependence. The design changes the whole training distribution, including digit and answer marginals; it does not isolate a pure carry-frequency mechanism.

Run from the repository root:

```bash
python3 src/difficulty_confound/export_scaling_covariates.py
```

The exporter uses only Python's standard library and reads the frozen artifacts without editing them. It writes:

- [data/features.csv](data/features.csv): 128 seed × condition × carry-stratum rows. Predictors derive only from task structure, configuration and actual training manifests.
- [data/outcomes.csv](data/outcomes.csv): matching held-out correct counts, accuracy and mean conditional predictive bits, kept in a separate file.
- [data/manifest.json](data/manifest.json): source hashes, permitted predictor roles, grouping metadata and checks. The 32 shuffled-label rows are controls and are excluded from the valid-label law fit.

The exporter verifies training counts, labels in valid-label conditions, unchanged test identities, train/test disjointness, exact correctness and distinct row keys. All rows reuse the same 4,000 distinct test questions; the independent training replication is eight paired seeds. No uncertainty is manufactured by treating 128 strata or 128,000 test predictions as independent trained models.

**Not identifiable here:** effects or exponents for model size, total training-set size or optimizer budget; all were held fixed. These are final addition accuracies, not RL improvements from an independently measured starting policy, so they must not be pooled as observations of $U$ in the existing RL fit. The exposure counts are compositional and sum to 16,000. Start with categorical carry strata: the observed accuracies are not monotonic in chain length, so an ordinal linear effect is an additional assumption. Any fit after seeing these published outcomes is exploratory. A held-out-seed check alone would not demonstrate extrapolation to a new exposure distribution or model scale.

The follow-up is now complete in [experiments/archive/02_heldout_exposure_difficulty_laws/](../../archive/02_heldout_exposure_difficulty_laws/). It crosses five widths, five mixtures and three checkpoints, withholding entire widths and mixtures from forecasting fits. The full extension reduced joint error 6.80% [-3.52, 14.80], missing its prospective criterion; starting difficulty alone reduced it 15.80% [10.27, 20.52] in a predeclared secondary comparison (95% paired seed bootstrap, eight seeds, 384 aggregate targets; p-val=n/a). It retains fixed addition questions but does not match all nuisance marginals. [Report and limitations](../../archive/02_heldout_exposure_difficulty_laws/RESULTS_heldout_laws.md).

## Where Kolmogorov complexity belongs

For the identical input, output and task specification, formal conditional Kolmogorov complexity is unchanged across exposure conditions. It is generally uncomputable, so assigning each item a measured “Kolmogorov complexity” would be unjustified. Use explicit program length under a fixed language and/or a fixed interpreter's execution count as operational structural features, with representation choices disclosed; these are not exact universal complexity.

For addition, the common algorithm is short even when its execution requires more carries. A program-length feature can consequently be constant and uninformative here. Keep computational work separate from description length. Model-dependent conditional predictive bits belong among outcomes or independent baseline-probe measurements, not among intrinsic complexity labels; they also exclude model/data coding costs. [Existing complexity note and primary sources](../../archive/01_fixed_width_addition_exposure/COMPLEXITY.md).

## Practical claim

The defensible bridge is: **apparent difficulty is a state of a task–model–training-history combination, and exposure-aware coordinates may improve predictive laws**. The addition result motivates testing that claim while keeping the task fixed. It does not establish that most benchmark difficulty is caused by data, that complexity disappears, or that the controlled result validates Varun's original language-model forecasts. The [actual-data companion](../../archive/02_heldout_exposure_difficulty_laws/RESULTS_actual_math.md) finds no improvement from its available breadth proxy; actual starting-difficulty and repeated-exposure measurements were unavailable.
