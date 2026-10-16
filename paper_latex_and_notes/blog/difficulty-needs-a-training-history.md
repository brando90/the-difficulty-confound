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

## What about Kolmogorov complexity?

The objection I expect is reasonable: complexity exists. A task's shortest description and its computational requirements cannot simply disappear because we changed the training set.

Kolmogorov complexity describes the length of a shortest program producing an object, relative to a fixed machine and any given information. Exact values are generally uncomputable. Description length also differs from execution time: a short program can run for a long time. [Grünwald and Vitányi's introduction](https://arxiv.org/abs/0809.2754) develops this distinction.

In our experiment, each test input, answer, task specification, and reference machine remains unchanged across exposure conditions. Their conditional description complexity is therefore unchanged. Addition already has a compact algorithm. Carry length is a structural property of one familiar addition procedure; we are not measuring Kolmogorov complexity by counting carries.

What if I train on data with high description complexity? With enough capacity, a network can memorize a finite collection of arbitrary labels; [Zhang and colleagues](https://arxiv.org/abs/1611.03530) demonstrated the practical importance of random-label fitting. But memorizing observed labels does not predict independent random labels on unseen examples. The information has to be supplied and stored somewhere. If it is in the weights, a total description-length account must pay for those weights.

Our loss in bits measures prediction quality given the trained model. It omits the cost of transmitting that model and its training data. It is not a full minimum description length measurement or an estimate of Kolmogorov complexity. [Voita and Titov](https://aclanthology.org/2020.emnlp-main.14/) provide practical coding approaches for a future study that charges for learning as well as prediction.

## The claim I would defend

Training alignment can reverse an apparent difficulty ranking while the measured task and test questions stay fixed. That is already a reason to be careful when interpreting benchmark failures as intrinsic task difficulty.

I would like to know whether data alignment explains more variation than structural complexity across a useful range of models and tasks. This addition study does not settle that broader claim. The answer will depend on the interventions, resource budgets, and definition of “more.” It also does not establish a reinforcement-learning result: all completed experiments here use supervised addition. Reinforcement learning is a motivation for extending the work.

The failures matter to the argument. Our earlier exposure study missed its strict no-harm criterion. An earlier predictive study missed its primary improvement criterion, and the small companion test on Varun's mathematics data got worse with a breadth proxy. The large ranking reversal is real within its setup, while reliable prediction remains something we must earn on fresh data.

For now, the sentence I want to carry forward is: **difficulty needs a training history**.

## Evidence and provenance

[Project results index](../../experiments/README.md) · [Sanmi argument and wording provenance](sanmi-argument-provenance.md) · [Detailed complexity note](../../experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md).

The numerical claims above come from committed experimental artifacts, not the cited external papers. This draft will include the fresh study's final result once verification finishes; the frozen experiment is not tuned to make the narrative succeed.
