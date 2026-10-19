# Experiment 15: Can measured alignment recover the true alignment?

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/15_alignment_alpha_recovery/README.md>

Experiment 15 scores every Experiment 11/12 training mix with the alignment-scaling paper's token-weighted measured alignment $\hat\alpha$. The task token is stripped before scoring. Measures: compression, probe embedding, their 50/50 combination, frozen gradient, and Task2Vec distance. In these mixes the true target share $\alpha$ was set by hand. The experiment tests whether $\hat\alpha$ ranks mixes by true $\alpha$ within every complexity level, whether it drifts with target complexity $K$ at fixed $\alpha$ (the difficulty-confound twist), and whether a calibrated $\hat\alpha$ can replace true $\alpha$ in the best law. A new 160-run arm, built from deterministic finite automata (DFAs) with rewired transitions, tests whether chance-corrected graded relevance predicts loss better than target share.

Status: DRAFT (10-18-2026), not run. Design: [PROTOCOL.md](PROTOCOL.md). Coding-agent prompt: [PROMPT.md](PROMPT.md). Program overview: [complexity versus alignment](../ideas/complexity_vs_alignment/README.md). Background note: [TASK2VEC_KOLMOGOROV.md](../ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md). Depends on [Experiment 14](../14_task2vec_vs_exact_complexity/README.md) (frozen probes, Task2Vec code).

**TLDR-end:** [tdc: Expt 15 readme] Test with numeric gates whether the alignment-scaling paper's measured $\alpha$ recovers the hand-set $\alpha$ of Experiments 11/12, stays flat across complexity, and can replace true $\alpha$ in the law, plus a graded-relevance training arm; CPU (central processing unit), about 2 h.
