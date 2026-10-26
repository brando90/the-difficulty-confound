# Experiment 10: Autoregressive Transformer Architecture Generalization

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/10_autoregressive_transformer_replication/README.md>

**TLDR:** Replicates Experiment 07's carry-marginal matched heterogeneous pretraining, balanced remediation, and prequential (online) MDL code length tracking on an **autoregressive causal transformer** (`AdditionTransformer`, 2 layers, 4 attention heads, causal self-attention, generating digits token-by-token from most to least significant).

- [PROTOCOL.md](PROTOCOL.md): Prospective research questions, functional forms, and frozen pass/fail gates.
- [RESULTS_transformer_history.md](RESULTS_transformer_history.md): Full analysis report, 3-panel figure, and prospective gate outcomes.
- [expt_v1/](expt_v1/): Code and scripts.

## Key Outcomes

- **Replay Verification:** `PASS` (all 200 final and 200 warm models replayed with greedy generation; `max_replay_bits_difference = 0.000e+00`).
- **Architecture Difference (Ceiling Saturation):** Unlike feedforward MLPs which maintain large headroom gaps across widths 32–384, autoregressive causal transformers learn digit addition so efficiently that widths 64, 96, 128, and 384 completely saturate at ceiling accuracy (>99.8% to 100.0%, loss < 0.005 bits) after 1,500 pretraining steps.
- **Width 32 Reversal:** At width 32, where ceiling saturation does not mask the effect, the compositional difficulty ranking reverses:
  - $q_{\text{pre}} = 0.10$: single-carry **99.50%** vs cascading two-carry **87.73%** (**+11.77 pp** gap).
  - $q_{\text{pre}} = 0.90$: single-carry **97.50%** vs cascading two-carry **99.88%** (**+2.38 pp** reversal).
  - Remediation ($q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$): cascading two-carry recovers **94.52%** of the gap (+8.49 pp to 99.34%) with **+0.23 pp** retention.
- **Prospective Gates:** Due to ceiling saturation across widths 64–384, pooled held-out loss reduction is 0.05% (Gate 1), prequential MDL reduction is 3.77% (Gate 2b), and pooled initial gap is 2.84 pp (Gates 3 & 4). All 5 gates are recorded as `NOT_ESTABLISHED` under the frozen criteria without post-hoc alterations.

