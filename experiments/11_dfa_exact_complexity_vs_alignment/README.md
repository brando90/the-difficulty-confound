# Experiment 11: Exact DFA complexity versus data alignment on regular languages

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/11_dfa_exact_complexity_vs_alignment/README.md>

This experiment trains a small transformer to accept or reject binary strings for target regular languages whose complexity is known exactly: the number of states $n \in \{4, 8, 16, 32\}$ in the minimal deterministic finite automaton (DFA), found with Hopcroft's algorithm. It crosses that complexity with the target's share of the training mix, $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$. It tests whether complexity raises the data bill (C1), whether alignment still dominates it in units of data (C2), and whether complexity moves the scaling-law coefficient or the exponent (C3).

Status: RUN, 10-18-2026. The protocol was frozen at commit 4caa44f1, 512 of 512 gated runs completed, and replay verification passed. Every gate is NOT_ESTABLISHED: see [RESULTS_dfa_complexity_alignment.md](RESULTS_dfa_complexity_alignment.md) and the dated log [CKPT_dfa_complexity_alignment.md](CKPT_dfa_complexity_alignment.md). The design is in [PROTOCOL.md](PROTOCOL.md). The self-contained coding-agent prompt is in [PROMPT.md](PROMPT.md). The program overview and the shared law family are in [the complexity-versus-alignment README](../ideas/complexity_vs_alignment/README.md).

**TLDR-end:** [tdc: Expt 11 README] Experiment 11 crosses exact minimal-DFA size with hand-set target share on a transformer sized for a CPU (central processing unit) to test whether complexity costs data, whether alignment dominates, and which scaling law holds; it has run, and every gate is NOT_ESTABLISHED because the transformer barely learned within the CPU budget.
