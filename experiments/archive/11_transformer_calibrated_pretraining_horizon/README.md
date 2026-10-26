# Experiment 11: Calibrated Pretraining Horizon for Causal Autoregressive Transformers

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/11_transformer_calibrated_pretraining_horizon/README.md>

**TLDR:** Calibrates the pretraining horizon of causal autoregressive transformers (`AdditionTransformer`, 2 layers, 4 heads) from 1,500 to 300 steps ($T_{\text{pre}}=300$) to match the active learning regime of MLPs, restoring active headroom across all five widths ($d_{\text{model}} \in \{32, 64, 128, 96, 384\}$) and evaluating whether starting-difficulty scaling laws predict held-out loss and prequential MDL on transformers.

- [PROTOCOL.md](PROTOCOL.md): Prospective research questions, functional forms, and frozen pass/fail gates.
- [RESULTS_transformer_calibrated.md](RESULTS_transformer_calibrated.md): Full analysis report, 3-panel figure, and prospective gate outcomes.
- [expt_v1/](expt_v1/): Code and execution scripts.

## Key Outcomes

- **Replay Verification:** `PASS` (all 200 final and 200 warm models replayed with greedy generation; `max_replay_bits_difference = 0.000e+00`).
- **Gate 3 (Pure Compositional Reversal):** **`SUPPORTED`**. At step 0, single-carries lead by **15.67 [12.13, 19.10] pp** at $q_{\text{pre}}=0.10$ ($83.99\%$ vs $68.32\%$), and reverse to cascading two-carries leading by **15.70 [12.89, 18.17] pp** at $q_{\text{pre}}=0.90$ across all 5 model scales.
- **Gate 2b (Prequential Online MDL Prediction):** **`SUPPORTED`**. Adding probed starting difficulty (`stable_difficulty` vs `stable_base`) reduces joint held-out prequential adaptation MDL RMSE by **18.84% [5.88, 34.16]** (`0.1626 -> 0.1320` bits), establishing that starting difficulty predicts online learning costs on causal transformers.
- **Compositional Remediation & No-Harm Retention:** Under balanced post-training ($q_{\text{pre}}=0.10 \to q_{\text{post}}=0.50$), cascading two-carry accuracy rises from $68.32\%$ to **$99.90\%$** (**+31.58 pp** gain), while single-carry control accuracy rises from $83.99\%$ to **$99.93\%$** (**+15.94 pp** retention gain). Gate 4 lower bound is 12.13 pp vs 15.0 pp line (`NOT_ESTABLISHED`).
- **Gate 1 (Conditional Loss Prediction):** `NOT_ESTABLISHED` (0.99% [-11.40, 5.66] reduction; 1,000 steps of balanced post-training compressed final test loss to 0.02 bits across all widths, whereas online prequential MDL accumulated the full adaptation gap).

