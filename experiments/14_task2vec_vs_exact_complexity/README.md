# Experiment 14: Does Task2Vec track exact task complexity?

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/14_task2vec_vs_exact_complexity/README.md>

Experiment 14 tests whether the Task2Vec Fisher-embedding norm tracks exactly computed task complexity $K$ (minimal deterministic finite automaton (DFA) states from Experiment 11; shortest domain-specific language (DSL) program from Experiment 12). It checks that the link survives controls for label entropy and sensitivity, whether the norm predicts how much data a task needs, and whether Brando's diversity coefficient measures coverage rather than complexity. It trains only small probe networks on a CPU (central processing unit) and reuses all Experiment 11/12 data. The question is whether Task2Vec can stand in for $K$ on real data.

Status: DRAFT (10-18-2026), not run. Design: [PROTOCOL.md](PROTOCOL.md). Coding-agent prompt: [PROMPT.md](PROMPT.md). Program overview: [complexity versus alignment](../ideas/complexity_vs_alignment/README.md). Background note: [TASK2VEC_KOLMOGOROV.md](../ideas/complexity_vs_alignment/TASK2VEC_KOLMOGOROV.md). Companion: [Experiment 15](../15_alignment_alpha_recovery/README.md) (recovering $\alpha$).

**TLDR-end:** [tdc: Expt 14 readme] Embed every Experiment 11/12 task with Task2Vec under frozen probes and test, with numeric gates, whether the embedding norm tracks exact complexity and data cost, and whether the diversity coefficient tracks coverage instead; CPU, about 1–2 h.
