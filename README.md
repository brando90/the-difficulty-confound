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

Both experiments are complete and archived. These results support exposure dependence and a candidate starting-difficulty covariate in controlled addition; they do not establish that difficulty is mostly data across language models.

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

Both controlled experiments are complete and archived with full artifacts. The first shows a large exposure effect while missing its control-performance retention criterion; the second misses its primary predictive-improvement criterion, with a promising secondary difficulty-only result.

## What stays fixed—and what does not

The operational construct is exact fixed-width addition. Input/output lengths, notation, target function, evaluator, and test items stay fixed within each experiment. Exposure comparisons at each width and checkpoint hold architecture, optimizer, and training budget fixed. The predictive-law study deliberately varies widths and budgets to measure transfer. The learned model changes: changing what it knows is the intervention. This does not prove that latent psychological skills or strategies are identical across conditions.

A positive result would establish a causal effect of the specified training-distribution intervention in this synthetic setting. It would not establish that most difficulty across language-model benchmarks is caused by data, nor that training exposure is the only explanation for a particular proprietary model's failures.

## Complexity belongs in the argument

Kolmogorov complexity is the length of the shortest program producing an object, relative to a fixed universal machine. It is not generally computable. For an unchanged question, answer, and task specification, its formal conditional complexity stays unchanged across our training conditions, even when model accuracy changes.

We measure **conditional predictive code length**: how many bits the trained model would need to encode the correct answer given the input. This is a computable model-dependent quantity; it excludes the cost of encoding the model and training data, so it is neither exact Kolmogorov complexity nor a complete minimum-description-length analysis. [Theory and reviewer-facing limitations](experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md).

## Connection to predictive scaling laws

Difficulty can enter a predictive law as a property of the task, starting model, and training history. Our [connection to the projects with Varun](experiments/ideas/predictability_bridge/README.md) separates structural demand, known training exposure, and difficulty measured at a particular checkpoint. It specifies extensions to the existing environment-utility model, leakage controls, and held-out-scale comparisons. The first experiment supplies separate feature and outcome tables but cannot identify scale exponents. The second prospectively tests an adaptation of the alignment-law form on unseen widths and exposure mixtures. The actual-data companion tests a breadth proxy; neither result validates the distinct reinforcement-learning utility law.

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
