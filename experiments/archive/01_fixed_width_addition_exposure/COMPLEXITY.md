# Complexity and the difficulty confound

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/01_fixed_width_addition_exposure/COMPLEXITY.md>

**TLDR:** Keep intrinsic task descriptions fixed across exposure interventions and measure learner-dependent predictability separately. Kolmogorov complexity is a conceptual invariant here, not a numerical estimator or a runtime bound.

## What Kolmogorov complexity contributes

For a fixed universal machine U, K_U(y | x, s) is the length of the shortest program producing answer y given input x and task specification s. Our paired conditions evaluate identical x, y, s and U, so this quantity is identical across conditions. We do not need to compute its value to establish that invariance.

For addition, there is a short program that computes every answer from the inputs. Changing the training set does not change this algorithm. Thus a large empirical accuracy change on the same items would demonstrate that algorithmic description complexity alone cannot determine this learner's success under a fixed training budget.

Conditioning additionally on training data D or learned weights W changes the question to K_U(y | x, s, D) or K_U(y | x, s, W). Such learner-relative information is relevant, but we cannot infer its value from accuracy alone.

## What this does not resolve

- Exact Kolmogorov complexity is generally uncomputable and depends on the reference machine up to an additive constant. Source-file size, gzip length, carry count, and neural-network parameter count are not exact estimates of it.
- A short program may take a long time to run. Description length does not remove computational complexity, search cost, or resource constraints. A future study could explicitly fix an interpreter and measure runtime or use time-bounded description length, with the machine and budget stated.
- The whole addition function has a short description. That says little about how many gradient updates a particular network needs to discover it.
- A fixed task and scorer support an operationally stable construct. They do not establish that the learned representations, strategies, or all component skills remain unchanged.

## The empirical information measure

For every held-out input, record the sum over four output digits of -log2 p(correct digit | input, trained model). This is a conditional predictive code length in bits per answer under the model's factorized distribution. Lower values mean the trained model predicts the correct answer more efficiently, including distinctions hidden by exact-match accuracy.

The code assumes the receiver already has the model and inputs. It omits model and training-data transmission costs. Therefore it is **not** an estimate of Kolmogorov complexity, a full minimum description length (MDL) score, or a prequential/online MDL experiment. A full MDL follow-up would charge for the model or transmit labels sequentially with a fixed learning rule and order; choosing that code after seeing results would need separate reporting.

## A defensible reviewer-facing claim

“On the same held-out addition items, a controlled change in training exposure changed performance while the target function and formal task descriptions remained fixed. This establishes exposure dependence for this learner and setup; it neither eliminates computational complexity nor estimates the share of real-world benchmark difficulty explained by data.”

Append “most of the prespecified baseline gap was recovered” only if the locked criterion passes, and state the measured fraction and interval. Do not substitute “difficulty is mostly data” for this bounded claim.

## Sources and relation to prior work

- [Kolmogorov, Three Approaches to the Quantitative Definition of Information](https://doi.org/10.1080/00207166808803030): the foundational conditional-description perspective. The original work dates to 1965; this identifier points to the 1968 English journal publication. An accessible [1965 paper copy](https://cqi.inf.usi.ch/qic/Kolmogorov_Complexity_1965.pdf) is also available.
- [Rissanen, Modeling by Shortest Data Description (1978)](https://doi.org/10.1016/0005-1098(78)90005-5): model selection through total description length.
- [Voita and Titov (2020)](https://aclanthology.org/2020.emnlp-main.14/): operational online and variational description-length methods for probing; useful for a full MDL follow-up.
- [Lake and Baroni (2018)](https://proceedings.mlr.press/v80/lake18a.html): command-to-action compositional generalization changes with the train/test relation.
- [Keysers et al. (2020)](https://arxiv.org/abs/1912.09713): distinguishes matching primitive distributions from matching their compositions, motivating stronger future controls.

These works establish substantial prior art. The initial experiment is a reproducible mechanism demonstration, not a claim of novelty or a complete paper.
