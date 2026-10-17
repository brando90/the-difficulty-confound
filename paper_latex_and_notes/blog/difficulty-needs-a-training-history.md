# Difficulty needs a training history

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/paper_latex_and_notes/blog/difficulty-needs-a-training-history.md>

**TLDR:** Changing training exposure can reverse which addition problems a neural network finds harder, even with the answers and task held fixed. Turning that observation into reliable performance forecasts is a separate problem, and our first attempt failed badly.

Draft by Brando Miranda, assisted by Codex. Updated 10-16-2026. [Google companion](https://docs.google.com/document/d/19yNEZTimoCPr9H4D_P-53nN83dBNFz_LY-aFdRa6_Bk/edit). This is a draft in the research repository, not a published personal-site post.

## The question that bothered me

This project started with a conversation with Sanmi about increasingly difficult benchmarks. He asked how we could raise difficulty while remaining confident that we were measuring the same thing. That question gets harder when a benchmark combines information gathering, reasoning, planning, and reporting: a lower score could reflect a different bottleneck.

My response was more basic. Models are trained on data. If we train them extensively on the supposedly difficult distribution, it may become familiar. A benchmark score therefore mixes the demands of the task with the learner's preparation for it.

I later put the intuition provocatively: “there isn't such a thing as difficulty—only alignment matters.” That is stronger than the evidence supports. The useful hypothesis is that **observed difficulty depends substantially on the relationship between training and evaluation**. Calling a task difficult without naming the learner, training history, and resource budget leaves something essential unspecified.

I wanted an experiment small enough that we could inspect this relationship directly.

## Keep addition fixed; change what the model practices

We train small neural networks from scratch to add two three-digit integers. Every answer has an exact checker. Within each comparison, the network architecture, initialization, training budget, and test questions stay fixed. We change how much training exposure goes to questions with different carry patterns.

The familiar intuition is that carrying makes addition harder. We compare questions requiring no carries with questions containing a chain of two carries. Crucially, we match the answer distribution: each training condition contains exactly 40 pairs for each of the same 117 sums. The two evaluation groups also have identical answer histograms. A model cannot explain the contrast merely by learning that one condition produces different answers more often.

In [experiments/archive/03_answer_matched_alignment/](../../experiments/archive/03_answer_matched_alignment/RESULTS_matched_alignment.md), training mostly on no-carry questions gives them a 25.30-point accuracy advantage [23.14, 27.26]. Training mostly on two-carry questions reverses the ranking: they lead by 24.58 points [22.88, 26.19]. The direction reverses at every tested network width.

Those are percentage points, averaged across five widths within each of eight paired seeds. The 95% intervals resample whole seeds and condition on the fixed test questions; p-val=n/a. Each group has 936 test questions. Both directions pass our predeclared 10-point threshold.

The operational skill is still exact addition. The same questions switch relative difficulty when their training history changes. We have not established that the networks use identical internal algorithms. Input-digit frequencies also differ across conditions, so this experiment identifies an effect of the training distribution, not a uniquely isolated psychological effect of carrying.

## The predictive law that blew up

This connects to my predictive-scaling work with Varun. If training alignment changes performance, perhaps a measure of exposure or starting difficulty can improve forecasts for models and data mixtures we have not fitted on.

We used starting loss on a separate probe set, measured before the intervention, as a learner-relative difficulty feature. Lower loss means the warm-started model already assigns more probability to the right answers. A simple positive predictive law combined model size, training updates, carry group, exposure fraction, and this starting loss.

The result was a useful failure. Adding starting difficulty increased joint held-out root mean squared error from 0.4727 [0.3212, 0.5990] to 292.7941 [0.3588, 507.0797] bits per answer, across 192 targets per law and eight seeds. These are seed-bootstrap 95% intervals; p-val=n/a. We kept every forecast.

One fitted seed assigned starting loss an exponent of about −7.20. Its fitting examples had starting losses between 2.53 and 3.92 bits, while a held-out model reached 0.51 bits. The fitted formula interpreted this better starting point as a reason to forecast enormously worse performance. Its largest forecast was 2,737.33 bits for an observed loss of 0.0932 bits. That seed contributed 99.97% of the joint squared error.

Exponentiating a regression guarantees positive forecasts. It does not guarantee sensible extrapolation. Model size and starting loss can partly substitute for one another inside the fitting range, leaving coefficients that behave badly outside it.

Our fourth study, [experiments/archive/04_monotone_ridge_heldout_addition/](../../experiments/archive/04_monotone_ridge_heldout_addition/RESULTS_stable.md), froze a repair before collecting 200 fresh trajectories: shrink the fitted coefficients and constrain their directions so starting-difficulty elasticity lies in $[0,1]$ and larger models, more updates, and more relevant exposure cannot increase predicted loss when other features are fixed.

The repair worked on fresh data (held-out widths 96 and 384, held-out mixtures 0.2 and 0.8): `stable_difficulty` reduced joint held-out root mean squared error from $1.94 \times 10^{12}$ bits per answer under the unconstrained difficulty law to $0.4835$ $[0.3760, 0.5780]$ bits per answer ($100.00\%$ $[99.90, 100.00]$ reduction), while replicating the ranking reversal ($24.07$ $[22.80, 25.46]$ and $24.48$ $[22.77, 26.06]$ percentage points). However, relative to an equally constrained baseline without starting difficulty (`stable_base`, $0.5010$ $[0.3894, 0.5972]$ bits), adding starting difficulty reduced joint error by only $3.48\%$ $[1.12, 6.24]$—statistically positive, but below our $10\%$ practical threshold. Why? Because every starting model in that experiment shared the exact same balanced 500-step warmup, leaving starting difficulty nearly redundant with model width and carry group.

## When starting difficulty earns its keep: heterogeneous training history and no-harm remediation

That diagnosis immediately suggested a direct test, which we froze and executed in [experiments/archive/05_pretraining_history_and_retention/](../../experiments/archive/05_pretraining_history_and_retention/RESULTS_history_retention.md). In realistic benchmark settings, we evaluate checkpoints whose prior training exposure is heterogeneous or unknown. We therefore pretrained 200 checkpoints across five carry mixtures ($q_{\text{pre}} \in \{0.1, 0.5, 0.9\}$ for fitting; $\{0.2, 0.8\}$ held out) and five network widths (`32/64/128` for fitting; `96/384` held out), probed each checkpoint's starting loss $D_{m_0}$ on an independent 936-item probe set, and then post-trained every checkpoint on a balanced remediation mixture ($q_{\text{post}}=0.50$).

Once starting checkpoints differed in their prior training history, probed starting difficulty became essential: `stable_difficulty` lowered joint held-out post-training RMSE from $0.1902$ $[0.1425, 0.2323]$ bits per answer under `stable_base` to $0.1004$ $[0.0688, 0.1268]$ bits per answer—a **$47.20\%$ $[42.88, 53.08]$ reduction** across 192 held-out targets with zero invalid forecasts, improving at both held-out width 96 ($47.21\%$) and width 384 ($24.59\%$). Remarkably, probed starting difficulty even outperformed an oracle law given the true log pretraining exposure fraction (`stable_oracle_history`, $0.1580$ $[0.1083, 0.1991]$ bits per answer) by **$36.43\%$ $[32.50, 39.58]$**, because a direct behavioral probe captures how much of the prior training distribution each specific model width actually absorbed.

The same experiment also resolved our earlier retention failure from Experiment 01. Checkpoints pretrained with scarce two-carry examples ($q_{\text{pre}}=0.10$) started with a $39.01$-point $[36.31, 41.57]$ accuracy gap between no-carry ($95.55\%$) and two-carry ($56.54\%$) items. Balanced post-training ($q_{\text{post}}=0.50$) raised two-carry accuracy to $90.70\%$ $[88.79, 92.43]$—recovering **$87.55\%$ $[83.60, 90.72]$** of the initial gap—while control no-carry accuracy *increased* by **$+1.09$ $[+0.33, +1.89]$ percentage points** to $96.64\%$, cleanly passing the strict $\ge -5.00$-point no-harm gate across all five widths.

## Isolating compositional carry exposure from per-column digit marginals

One sharp methodological objection still remained in Experiments 03–05: even when exact sums are matched, no-carry pairs `(0,0)` require column digit sums below 10, whereas two-carry pairs `(1,1)` require column digit sums of 10 or more. Could the ranking reversal merely reflect per-column digit marginals or single-column carry rates rather than multi-step carry composition?

In [experiments/archive/06_compositional_carry_marginal_matched/](../../experiments/archive/06_compositional_carry_marginal_matched/RESULTS_compositional.md), we matched both exact answer histograms and exact per-column carry marginals across every training mixture $q \in \{0.1, 0.2, 0.5, 0.8, 0.9\}$. We contrasted an **Isolated Single-Carry Regime** ($50\%$ ones-only carry `(1,0)` + $50\%$ tens-only carry `(0,1)`) against a **Paired Zero/Double-Carry Regime** ($50\%$ no-carry `(0,0)` + $50\%$ cascading two-carry `(1,1)`). Every training mixture had exactly $50.00\%$ ones-column carries (`3600/7200`) and $50.00\%$ tens-column carries (`3600/7200`), reducing the input-digit marginal Total Variation distance by more than $11\times$ (from a mean of $0.443$ in Experiments 03–04 down to $0.0398$).

Even with identical answer histograms and identical $50\%/50\%$ per-column carry frequencies, the difficulty ranking between isolated single carries and cascading two-carry chains reversed at every width: single carries led by **$22.35$ $[20.60, 24.00]$ percentage points** ($93.70\%$ vs $71.35\%$) at $q=0.10$, while cascading two-carry chains led by **$17.74$ $[14.62, 21.12]$ percentage points** ($92.13\%$ vs $74.39\%$) at $q=0.90$. And because Experiment 06 intentionally returned to a shared balanced warmup ($q=0.50$), starting difficulty once again added only $1.94\%$ $[-3.10, 6.50]$ incremental RMSE reduction—reinforcing the exact contrast with Experiment 05's $47.20\%$ gain under heterogeneous histories.

## What about Kolmogorov complexity?

The objection I expect is reasonable: complexity exists. A task's shortest description and its computational requirements cannot simply disappear because we changed the training set.

Kolmogorov complexity describes the length of a shortest program producing an object, relative to a fixed machine and any given information. Exact values are generally uncomputable. Description length also differs from execution time: a short program can run for a long time. [Grünwald and Vitányi's introduction](https://arxiv.org/abs/0809.2754) develops this distinction.

In our experiment, each test input, answer, task specification, and reference machine remains unchanged across exposure conditions. Their conditional description complexity is therefore unchanged. Addition already has a compact algorithm. Carry length is a structural property of one familiar addition procedure; we are not measuring Kolmogorov complexity by counting carries.

What if I train on data with high description complexity? With enough capacity, a network can memorize a finite collection of arbitrary labels; [Zhang and colleagues](https://arxiv.org/abs/1611.03530) demonstrated the practical importance of random-label fitting. But memorizing observed labels does not predict independent random labels on unseen examples. The information has to be supplied and stored somewhere. If it is in the weights, a total description-length account must pay for those weights.

Our loss in bits measures prediction quality given the trained model. It omits the cost of transmitting that model and its training data. It is not a full minimum description length measurement or an estimate of Kolmogorov complexity. [Voita and Titov](https://aclanthology.org/2020.emnlp-main.14/) provide practical coding approaches that charge for learning as well as prediction.

In [experiments/archive/07_prequential_mdl_and_length_transfer/](../../experiments/archive/07_prequential_mdl_and_length_transfer/RESULTS_mdl_history.md), we directly measured **prequential (online) minimum description length** along a 1,000-step balanced adaptation trajectory while holding exact $50\%/50\%$ per-column carry rates invariant across heterogeneous pretraining histories ($q_{\text{pre}} \in \{0.1, 0.2, 0.5, 0.8, 0.9\}$). Online adaptation code length reversed across all five widths between $q_{\text{pre}}=0.10$ ($0.4423$ $[0.4043, 0.4824]$ bits per item) and $q_{\text{pre}}=0.90$ ($0.1773$ $[0.1525, 0.2016]$ bits per item), and adding probed starting difficulty reduced joint held-out conditional loss RMSE by **$36.22\%$ $[32.45, 40.51]$** and joint held-out prequential online MDL RMSE by **$56.93\%$ $[55.01, 58.67]$**.

## Extending to verifiable-reward reinforcement learning (RLVR)

Finally, in [experiments/archive/08_rlvr_verifiable_reward_utility_laws/](../../experiments/archive/08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md), we replaced supervised post-training with **Reinforcement Learning with Verifiable 0/1 Exact-Sum Rewards (RLVR / leave-one-out GRPO, $G=16$ rollouts per prompt)** across 200 pretrained checkpoints. Three results stood out:

1. **RL improvement utility reverses with pretraining history:** Realized RL accuracy gain $U = \text{Acc}_T - \text{Acc}_0$ reversed at every width (`+29.07 [26.32, 32.33] pp` advantage for $S_{\text{comp}}$ at $q_{\text{pre}}=0.10$ vs `+32.70 [30.18, 35.20] pp` advantage for $S_{\text{iso}}$ at $q_{\text{pre}}=0.90$), while `stable_difficulty` reduced held-out post-RLVR loss RMSE by **$39.87\%$ $[36.20, 50.17]$**.
2. **Pure 0/1 RLVR achieves natural no-harm retention:** Pure 0/1 exact-sum RLVR recovered **$78.23\%$ $[72.58, 84.23]$** of the initial pure compositional gap (`+35.49 pp` on cascading two-carry `(1,1)`), while single-carry control accuracy *improved* by **$+6.45$ $[+4.69, +7.91]$ percentage points** (`77.81% → 84.26%`), because group-relative RL assigns zero advantage ($\hat{A}_{b,g}=0$) whenever all $G=16$ rollouts on an already-mastered item succeed.
3. **Why additive linear RL utility laws need a headroom ceiling:** At interpolated held-out width 96, adding stochastic rollout signal density $\log \rho_{m_0}$, headroom $H_{m_0}$, and $\log D_{m_0}$ cut RL utility RMSE by **$48.56\%$ $[41.65, 57.30]$** (`14.59 → 7.50` pp, beating oracle history by **$12.01\%$**). At extrapolated near-ceiling width 384 (where $\text{Acc}_0$ is already $78\%\text{–}99.8\%$), however, the *additive linear* utility form from `rl-4-code-verification` overshot remaining headroom $1 - \text{Acc}_0$ (`19.71 → 27.38` pp), pinpointing the need for a headroom-bounded RL utility law.

## The claim I would defend

Training alignment can reverse an apparent difficulty ranking—across final accuracy, prequential online MDL code length, and verifiable-reward RL improvement utility—while the measured task, exact answer distribution, and even per-column carry frequencies stay fixed. Moreover, when models arrive with different prior training histories, an independent starting-difficulty probe substantially improves held-out post-training forecasts ($36.22\%\text{–}47.20\%$ loss RMSE reduction; $56.93\%$ prequential MDL RMSE reduction) and guides targeted supervised or RLVR remediation that closes the difficulty gap without harming existing capabilities.

I would like to know whether data alignment explains more variation than structural complexity across a broader range of models and tasks. This addition series does not settle that broader claim for general language models.

The negative results along the way sharpen the conclusion. Experiment 01 showed that naive distribution shifts can hurt unpracticed strata; Experiment 03 showed that unconstrained log-linear difficulty laws can blow up out of range; Experiments 04 and 06 showed that probing starting difficulty adds little when all checkpoints share the exact same warmup history; and Experiment 08 showed that additive linear RL utility laws overshoot remaining headroom on near-ceiling larger models even when multiplicative log-loss laws succeed.

Together, the eight experiments give concrete empirical meaning to the thesis: **difficulty needs a training history**.

## Evidence and provenance

[Project results index](../../experiments/README.md) · [Sanmi argument and wording provenance](sanmi-argument-provenance.md) · [Detailed complexity note](../../experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md).

The numerical claims above come from committed, SHA-256-verified experimental artifacts across Experiments 01–08, not the cited external papers.
