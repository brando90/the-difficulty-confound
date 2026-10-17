# Compositional carry exposure with invariant per-column carry rates and matched answers

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/06_compositional_carry_marginal_matched/PROTOCOL.md>

**TLDR:** Test whether apparent difficulty rankings between isolated single carries (`(1,0)`, `(0,1)`) and composed cascading two-carry chains (`(1,1)`) reverse when **both** exact answer histograms **and** per-column marginal carry rates ($50\%$ at ones, $50\%$ at tens, $0\%$ at hundreds) are held strictly invariant across every training mixture.

Frozen on 10-16-2026 before training.

## Fixed skill and compositional carry matching

In 3-digit addition with totals below 1000 ($c_2=0$), every pair falls into one of four carry-vector cells $(c_0, c_1) \in \{(0,0), (1,0), (0,1), (1,1)\}$:
- **Isolated Single-Carry Regime ($S_{\text{iso}}$, stratum 1):** 50% ones-only carry `(1,0)` + 50% tens-only carry `(0,1)`.
- **Paired Zero/Double-Carry Regime ($S_{\text{comp}}$, stratum 2):** 50% no-carry `(0,0)` + 50% cascading two-carry chain `(1,1)`.

Select all 180 eligible sums (<1000) with at least 28 unordered pairs in every one of the four cells. Reserve 4 test, 2 calibration, and 2 probe items per `(answer, cell)` (2,880 test, 1,440 calibration, 1,440 probe items total). Sample $20(1-q)$ training pairs from `(1,0)` and `(0,1)` each, and $20q$ training pairs from `(0,0)` and `(1,1)` each (7,200 training pairs per mixture, 40 per answer). For every $q \in \{.1, .2, .5, .8, .9\}$, exact ones-column carry rate is $3600/7200 = 50.00\%$ and exact tens-column carry rate is $3600/7200 = 50.00\%$.

- **Fresh seeds:** `6101, 6202, 6303, 6404, 6505, 6606, 6707, 6808`.
- **Widths and checkpoints:** Fit widths `32/64/128`, held-out widths `96/384`; 500 balanced warmup updates ($q=0.50$), intervention checkpoints `300/1000/3000`.
- **Primary compositional gate:** Pure isolated single-carry `(1,0)+(0,1)` vs composed double-carry `(1,1)` advantage must exceed 10 percentage points in both directions ($q=0.1$ and $q=0.9$), with positive 95% seed-bootstrap lower bounds and positive reversal at all five widths. Regime-level $S_{\text{iso}}$ vs $S_{\text{comp}}$ contrast (which averages `(1,1)` with `(0,0)`) uses a 5-point gate.
