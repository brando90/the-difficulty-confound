# Experiment 16: Addition crossed, carry-chain length versus alignment

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/16_carry_alignment_crossed_arithmetic/README.md>

Experiment 16 brings the complexity-versus-alignment program back to addition. It crosses carry-chain length with the target stratum's training share $\alpha$ and fits the program's law family. Carry-chain length is a structural demand, not Kolmogorov complexity; the law variable is $s = c + 1$. The experiment tests whether chain length costs data (C1), whether alignment dominates it (C2), and whether it moves the coefficient or the exponent the way exact complexity does in Experiments 11 and 12 (C3). There are two arms:

- **16b (primary, GPU = graphics processing unit):** fresh 8-digit problems, three transformer sizes, plus a reversed-output control. The control tests whether the cost exists only when the output format makes the chain sequential. About 90 A100-hours, launched from [`launch/`](launch/README.md).
- **16a (secondary, CPU = central processing unit):** the answer-matched 3-digit multilayer perceptron (MLP) setup of Experiments 06 and 08. About 25 min.

Status: DRAFT (10-18-2026), not run. Design: [PROTOCOL.md](PROTOCOL.md). Coding-agent prompt: [PROMPT.md](PROMPT.md). Program overview: [complexity versus alignment](../ideas/complexity_vs_alignment/README.md).

**TLDR-end:** [tdc: Expt 16 readme] Cross carry-chain length (a structural demand, not $K$) with target share on 8-digit transformer addition and 3-digit MLP addition, and test with numeric gates whether chain length costs data, whether alignment dominates, and whether it enters the law like exact complexity; GPU primary arm plus a short CPU arm.
