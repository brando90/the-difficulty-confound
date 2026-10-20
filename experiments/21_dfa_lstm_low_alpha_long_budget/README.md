# Experiment 21: Floor or late learning? Experiment 20's low-share LSTM cells run 4× longer

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/21_dfa_lstm_low_alpha_long_budget/README.md>

In [Experiment 20](../20_dfa_lstm_complexity_vs_alignment/README.md) every LSTM run at target share $\alpha \in \{0.05, 0.15\}$ stalled at 0.85–1.0 bits by $2^{16}$ examples, so its alignment-floor gate (G6) passed, but a floor inside the budget cannot be told from learning that starts later. This experiment reruns those 256 runs (same 16 target languages, data rule, LSTM, optimizer and 8 seeds) to $2^{18}$ examples and decides, run by run, whether the loss drops after $2^{16}$.

Status: RUN, 10-20-2026. The protocol was frozen at commit b2f1db7b, 256 of 256 runs completed, and replay verification passed. Overall verdict **LATE_LEARNING**: at α = 0.15 the n = 4 and n = 8 runs fall from 0.84 and 0.94 bits at 2^16 to 0.09 and 0.27 bits at 2^18 (29 and 22 of 32 runs cross 0.5 bits), so Experiment 20's floor was a budget artifact there; n = 16 at α = 0.15 and n ≤ 16 at α = 0.05 are UNRESOLVED, n = 32 FLOOR_HOLDS. See [RESULTS_dfa_lstm_long_budget.md](RESULTS_dfa_lstm_long_budget.md). Design and disclosure: [PROTOCOL.md](PROTOCOL.md). Dated log: [CKPT_dfa_lstm_long_budget.md](CKPT_dfa_lstm_long_budget.md).

**TLDR-end:** [tdc: Expt 21 README] Experiment 20's stalled low-share LSTM runs, rerun 4× longer: at α = 0.15 the small automata (n = 4, 8) learn late, so the alignment floor was a budget artifact; misalignment shows up as a long delay, not a floor.
