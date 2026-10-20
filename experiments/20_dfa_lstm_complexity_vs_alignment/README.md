# Experiment 20: Exact DFA complexity versus data alignment, with an LSTM learner

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/20_dfa_lstm_complexity_vs_alignment/README.md>

This experiment reruns [Experiment 11](../11_dfa_exact_complexity_vs_alignment/README.md) with the same tasks, data, mixes, seeds, checkpoints, analysis and gates. Only the learner changes: Experiment 11's 84,290-parameter one-layer LSTM replaces the 104,002-parameter transformer. The 16 target regular languages have exact minimal-DFA sizes $n \in \{4, 8, 16, 32\}$, and each is trained at target shares $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$. In Experiment 11 the transformer barely learned within $2^{16}$ examples, so every gate was NOT_ESTABLISHED. Its exploratory LSTM arm, by contrast, learned the $n \le 16$ languages.

Status: RUN, 10-20-2026. The protocol was frozen at commit d3ab980b, 512 of 512 gated runs completed, and replay verification passed. G1 (seen data), G3b and G6 are SUPPORTED; G2, G3a, G4, G5 and G7 are NOT_ESTABLISHED: see [RESULTS_dfa_complexity_alignment.md](RESULTS_dfa_complexity_alignment.md). Design and disclosure: [PROTOCOL.md](PROTOCOL.md). Dated log: [CKPT_dfa_lstm_complexity_alignment.md](CKPT_dfa_lstm_complexity_alignment.md).

**TLDR-end:** [tdc: Expt 20 README] Experiment 20 reruns Experiment 11's exact-DFA complexity-versus-alignment grid with an LSTM that can learn these languages. Everything else, the gates included, is held fixed, and the α = 1 slice was already seen in Experiment 11's exploratory arm.
