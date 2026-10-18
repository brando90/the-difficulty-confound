# Experiment 11: Calibrated Pretraining Horizon for Causal Autoregressive Transformers

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/11_transformer_calibrated_pretraining_horizon/README.md>

**TLDR:** Calibrates the pretraining horizon of causal autoregressive transformers (`AdditionTransformer`, 2 layers, 4 heads) from 1,500 to 300 steps ($T_{\text{pre}}=300$) to match the active learning regime of MLPs, restoring active headroom across all five widths ($d_{\text{model}} \in \{32, 64, 128, 96, 384\}$) and evaluating whether starting-difficulty scaling laws predict held-out loss and prequential MDL on transformers.

- [PROTOCOL.md](PROTOCOL.md): Prospective research questions, functional forms, and frozen pass/fail gates.
- [expt_v1/](expt_v1/): Code and execution scripts.
