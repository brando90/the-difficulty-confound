Run on 10-19/20-2026 on a shared 4-core CPU box. All 512 of 512 gated runs completed (8 seeds × 16 targets × 4 target shares α) in 8,231 s of grid wall time. The learner is the 84,290-parameter one-layer LSTM. Tasks, data, mixes, seeds, checkpoints, analysis code and gate thresholds are Experiment 11's, and the protocol was frozen at commit d3ab980b before any gated run. K is exact for the DFA description language (minimal-DFA size n, by Hopcroft and Moore), K_bits = 20, 56, 144, 352.

## Takeaway

**With a learner that does learn these languages, complexity is real and costs data. Alignment does not dominate: mixing in other tasks cost the LSTM far more than its share of the data. Three of the eight gates pass: G1 (seen data), G3b and G6.**

- **Complexity costs data (G1 SUPPORTED, not prospective).** At α = 1, reaching 0.5 bits takes 13.0 times as many examples for n = 32 as for n = 4 (seed interval 12.6 to 13.4; two-stage 7.5 to 20.6). That is a lower bound, because 31 of 32 n = 32 runs never reach 0.5 bits. These α = 1 runs repeat Experiment 11's exploratory arm, which had already been seen.
- **Complexity moves the exponent, not the coefficient (G3b SUPPORTED, prospective).** In M_K_both the learning-curve exponent falls by γ1 = −0.142 per doubling of K_bits (interval −0.146 to −0.137). The exponent-only law beats the coefficient-only law on held-out cells by 5.3% (interval 4.8% to 5.8%), just over the 5% bar. κ is negative (−1.17), so the description-length picture (G3a) fails. C3 reads as the exponent (quanta) picture. The margin over the bar is thin, and every law fits these curves poorly (held-out RMSE 1.2 to 1.9 in log2 bits, a typical miss of a factor of 2 to 4).
- **Target share sets a floor within this budget (G6 SUPPORTED, prospective).** Adding a1(1 − α) to the K law gives a1 = 0.84 bits (interval 0.83 to 0.85) and cuts held-out error by 23% (interval 21% to 25%). Every run at α = 0.05 and 0.15 is flagged as plateaued, at every n, and ends at 0.85 to 1.0 bits. A floor inside 2^16 examples cannot be told apart from learning that would start later.
- **Alignment does not dominate (G4 NOT_ESTABLISHED, prospective).** The elasticity ratio is E = 0.93 with K in bits (interval 0.91 to 0.94) and 0.67 in states, below the bar of E > 1. Per log unit, complexity costs a little more data than misalignment. The model-free E_mf is 1.96 at the 0.7-bit fallback, but 72% to 78% of its pairs are censored.
- **Distractors hurt (post hoc).** The effective-data exponent is η = 1.56 in the K law (interval 1.46 to 1.69), above 1, which reads as interference; M_align_eff alone gives 0.99. A direct check agrees. An n = 4 target at α = 0.4 has seen 26,214 target examples by 2^16 and sits at a median 0.49 bits. On its own, the same target is at 0.004 bits after 16,384 examples. The penalty shrinks as n grows, because large automata are not learned either way.
- **G2 NOT_ESTABLISHED** (the K law improves held-out error by 8.2% against the 10% bar), **G5 NOT_ESTABLISHED** (at α = 1 the n = 32 minus n = 4 gap grows from 0.36 to 0.67 bits, because n = 4 goes to about 0) and **G7 NOT_ESTABLISHED** (Spearman −0.17).

## What was prospective (PROTOCOL Section 2)

The 128 α = 1 cells reproduce Experiment 11's exploratory LSTM arm (verify step 10 below). Gates that use only α = 1 rows were therefore not prospective. Predictions were written in PROTOCOL Section 5 before any α < 1 run.

| Gate | Prospective? | Prediction | Verdict | Prediction right? |
|---|---|---|---|---|
| G1 complexity is real | no (α = 1 seen) | SUPPORTED | SUPPORTED | yes |
| G2 K adds value beyond alignment | yes | SUPPORTED | NOT_ESTABLISHED (8.2% < 10%) | no |
| G3a coefficient (MDL) | yes | unresolved | NOT_ESTABLISHED | (see G3b) |
| G3b exponent (quanta) | yes | unresolved | SUPPORTED | no |
| G4 alignment dominates | yes | NOT_ESTABLISHED, E < 1 | NOT_ESTABLISHED, E = 0.93 | yes |
| G5 gap shrinks | no (α = 1 seen) | NOT_ESTABLISHED | NOT_ESTABLISHED | yes |
| G6 alignment floor | yes | NOT_ESTABLISHED | SUPPORTED | no |
| G7 quanta mechanism | no (α = 1 seen) | not predicted | NOT_ESTABLISHED | n/a |
| η | yes | below 1 (transfer) | 1.56, above 1 (interference) | no |

## Experiment 20 (LSTM) next to Experiment 11 (transformer)

| Quantity | Exp 11 transformer | Exp 20 LSTM |
|---|---|---|
| G1 M_K at 0.5 bits | 1.55, NOT_ESTABLISHED | 13.0, SUPPORTED |
| G2 held-out gain of the K law | 7.2%, NOT_ESTABLISHED | 8.2%, NOT_ESTABLISHED |
| G3 γ1 per doubling of K_bits; exp-over-coef gain | −0.011; 2.6%, unresolved | −0.142; 5.3%, exponent (G3b) |
| G4 E (bits / states); E_mf | 0.083 / 0.060; 4.3 | 0.93 / 0.67; 1.96 |
| G5 ρ | 5.9 | 1.86 |
| G6 a1; held-out gain | 0.043 bits; 0.3% | 0.84 bits; 23%, SUPPORTED |
| G7 Spearman(γ_j, τ_j) | 0.01 | −0.17 |
| η (K law) | 0.71 (transfer) | 1.56 (interference) |

Mean ε(2^16) in bits by level and α (Exp 20 LSTM / Exp 11 transformer):

| n | α = 0.05 | α = 0.15 | α = 0.4 | α = 1 |
|---|---|---|---|---|
| 4 | 0.96 / 0.89 | 0.85 / 0.85 | 0.33 / 0.73 | 0.0004 / 0.54 |
| 8 | 1.00 / 0.97 | 0.94 / 0.95 | 0.50 / 0.89 | 0.003 / 0.73 |
| 16 | 1.01 / 0.97 | 0.97 / 0.97 | 0.66 / 0.96 | 0.13 / 0.88 |
| 32 | 1.00 / 0.98 | 0.99 / 0.98 | 0.95 / 0.98 | 0.67 / 0.97 |

Below the resolution floor (10^-3 bits): 1.9% of all checkpoint rows, all at α = 1 (21% of n = 4 rows and 9% of n = 8 rows there), so the censored residual matters only for the smallest automata. All three ε* levels are reported in the target table below. With the LSTM the n = 4 to 16 targets reach 0.1 bits at α = 1, so the 0.3- and 0.1-bit columns are informative too.

Post hoc effective-data check (not pre-registered; results/comparison.json): the median over seeds and targets of log2[ε(α, 2^16) / ε(α = 1, α·2^16)], both floored at 10^-3 bits. Positive values mean the mixed run is worse than its own target examples predict.

| n | α = 0.05 | α = 0.15 | α = 0.4 |
|---|---|---|---|
| 4 | 0.55 | 4.20 | 8.20 |
| 8 | 0.15 | 1.38 | 4.85 |
| 16 | 0.13 | 0.36 | 0.43 |
| 32 | 0.03 | 0.08 | 0.08 |

## Deviations from PROTOCOL.md

1. **Two chunks.** The grid ran as two resume-safe chunks: 394 cells in 6,626 s, then 118 cells in 1,605 s. The protocol allows any number of chunks, and both are logged in manifest.json and CKPT. No cell was rerun. The box was shared (load average 8 to 10 in the first hour), so early cells took about 170 s instead of the projected 59 s.
2. **One added script.** `compare_exp11.py` writes the reported comparisons (results/comparison.json). Its effective-data check is post hoc and labelled as such. No gate, threshold, split, seed or ε* changed, and analyze.py ran unmodified.
3. **Steps left to the parent task.** No push, README row or pull request was made in this run. The branch is committed locally.

## Limitations

- **The laws fit poorly.** Held-out RMSE is 1.22 to 1.89 in log2 bits for every law, against 0.18 to 0.21 for the transformer, whose curves barely moved. LSTM curves stay near chance and then drop by orders of magnitude within one or two octaves, which no single power law in T follows. The gate verdicts compare these imperfect laws under the frozen rules. G3b in particular passes by 0.3 points over its bar.
- **The floor and the interference are within budget.** At α ≤ 0.15 the target sees at most 9,830 examples, interleaved with 16 other tasks. Whether those runs would learn with more data is not tested here.
- **This is a recurrent learner.** LSTMs simulate DFAs natively, so these verdicts hold for this learner and need not carry over to transformers or language models (Experiment 19 reruns the transformer on GPU).
- **G4 sits near its crossover.** The law-based E rises steeply with ε*: 0.70 at 0.40 bits, 0.93 at the pre-registered 0.5 bits, 1.11 at 0.56 bits and 1.81 at 0.71 bits (analysis.json E.curve; figure panel c). Alignment dominates only for targets close to chance, so the G4 verdict holds at the frozen level and would flip just above it.
- **n = 32 is still mostly unlearned** (mean 0.67 bits at α = 1), so M_K and e_K are lower bounds.
- K is exact only for the DFA description language and this sampler. There is one model size, one learning rate and lengths 8 to 24, and the distractors are unrelated languages.
