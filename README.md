# The Difficulty Confound

**Is a task hard—or has a model simply not learned it yet?**

This project investigates the hypothesis that much of what we call task difficulty in AI evaluation is explained by training data: its coverage, frequency, and relevance to the evaluated task.

Difficulty may be real, but observed failure alone does not establish that a task is intrinsically difficult. It may also reflect unfamiliarity, missing prerequisites, or a mismatch between training and evaluation distributions.

## Research question

How much does measured task difficulty change with controlled training exposure, while the skill being evaluated stays fixed?

## Core hypothesis

Targeted training can substantially reduce apparent difficulty, including on held-out examples. If so, benchmark difficulty should be interpreted relative to a model's training history and evaluation conditions.

This is a hypothesis to test, not an assumed conclusion.

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

Research planning. Experiments and results are forthcoming.
