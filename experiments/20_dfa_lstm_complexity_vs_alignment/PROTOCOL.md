# Protocol: Exact DFA complexity versus data alignment, rerun with an LSTM learner (`20_dfa_lstm_complexity_vs_alignment`)

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/20_dfa_lstm_complexity_vs_alignment/PROTOCOL.md>

Status: Frozen on 10-19-2026 before any gated Experiment 20 training. The freeze commit is logged in [CKPT_dfa_lstm_complexity_alignment.md](CKPT_dfa_lstm_complexity_alignment.md). After the freeze nothing in Sections 2–5 changes; deviations go in RESULTS.

This is the pre-registered rerun of [Experiment 11](../11_dfa_exact_complexity_vs_alignment/PROTOCOL.md) (frozen at commit 4caa44f1) that Experiment 11's RESULTS proposed. Brando Miranda approved it on 10-18-2026 ("Go") after hearing the Experiment 11 result. Notation, claims and laws are those of Experiment 11 and of the [program overview](../ideas/complexity_vs_alignment/README.md).

## 1. Why this rerun

Experiment 11 asked three questions about random regular languages whose minimal-DFA size $n \in \{4, 8, 16, 32\}$ is exact: does complexity cost data (C1), does alignment still dominate in units of data (C2), and does complexity move the scaling coefficient or the exponent (C3)? Every gate came back `NOT_ESTABLISHED`, for one reason: the 104,002-parameter transformer barely learned within $2^{16}$ examples. All 256 runs at $n \ge 16$ stayed near chance and were flagged as plateaued, and 94% of the model-free $T^\star$ values were censored at the budget. The exploratory LSTM arm of the same experiment learned $n = 4$ to $16$ almost perfectly on the same data. So the data can answer the questions, and that learner did not.

Experiment 20 asks the same three questions with the same tasks, data, mixes, seeds, checkpoints, analysis and gates. Only the learner changes: Experiment 11's 84,290-parameter one-layer LSTM. The transformer at a longer budget is a separate experiment on GPU (Experiment 19), so it is not repeated here.

## 2. What was seen before this protocol was written (disclosure)

This protocol was written after reading all of the following:

- **Experiment 11's full results** for the transformer: every gate, law fit, held-out score, $T^\star$ table and plateau census ([RESULTS](../11_dfa_exact_complexity_vs_alignment/RESULTS_dfa_complexity_alignment.md)).
- **Experiment 11's exploratory LSTM arm**: 128 runs, $\alpha = 1$ only, all 16 targets, all 8 training seeds, the same data, schedule and optimizer, with `metrics.csv` rows at all 8 checkpoints. Its summary was read: mean $\varepsilon(2^{16})$ of 0.000, 0.003, 0.135 and 0.669 bits for $n = 4, 8, 16, 32$; geometric-mean $T^\star(0.1)$ of about 7.7k, 15.7k, 39k and 65k (31 of 32 censored at $n = 32$); and 97 of 128 runs reaching 0.5 bits.
- **The three medians in Section 2.6 below**, computed from the arm in this run to apply the $\varepsilon^\star$ rule.

Nothing else about the LSTM was seen. **No LSTM run at $\alpha \lt 1$ exists**, and no law has been fit and no held-out forecast made on LSTM data.

**Consequence for what is prospective.** Training is bitwise deterministic for a given seed, torch version and one thread, so the 128 gated cells at $\alpha = 1$ will reproduce the arm's runs exactly. `verify.py` checks this (step 10, Section 6). Their numbers are therefore already known. So:

- **Not prospective (verdicts reported, but the data were seen):** G1, G5 and G7, the per-target exponents $\gamma_j$, the Section 4.6 $K$-versus-sensitivity regression and the $\alpha = 1$ column of the plateau census. All of these use $\alpha = 1$ rows only.
- **Prospective:** G2, G3a, G3b, G4, G6, $\eta$ and every held-out score. They depend on the 384 unseen cells at $\alpha \in \{0.05, 0.15, 0.4\}$ (75% of the grid). Their law fits also use $\alpha = 1$ fit points, which were seen. Of the 260 held-out points per seed, 44 are at $\alpha = 1$ (32 on the $K = 16$ slice and 12 at $T = 2^{16}$), so 216 (83%) are unseen.

RESULTS must print this label next to each gate.

## 3. Design

### 3.1 Unchanged from Experiment 11

Everything in Experiment 11's PROTOCOL Sections 2.1–2.3 and 2.5 holds here word for word, at its frozen commit 4caa44f1:

- the description language and $K_{\text{bits}} = 20, 56, 144, 352$;
- the sampler, the filters, the covariates and the sensitivity-stratified selection of 16 targets and 16 distractors from task seed 20261011;
- the hash train/test partition, the shared 4,096-string test set and the 69,632-string pool per task;
- the mix construction: $\alpha \in \{0.05, 0.15, 0.4, 1.0\}$, the same 16 distractors in every condition, the exact octave schedule, single-length batches of 64, and no pool reuse;
- training seeds 11101 … 11808, analysis seed 29101126, 20,000 bootstrap draws;
- the grid of 4 levels × 4 targets × 4 $\alpha$ × 8 seeds = **512 runs** of 1,024 steps, with checkpoints at $T = 2^9 … 2^{16}$;
- the fit and held-out cells: $K \in \{4, 8, 32\}$ versus 16, $\alpha \in \{0.05, 0.4, 1\}$ versus 0.15, and $T \le 2^{15}$ versus $2^{16}$.

The code is Experiment 11's, copied into `expt_v1/`. `dfa.py`, `data.py`, `laws.py` and `independent_fit.py` are byte-identical. `analyze.py` differs only in its docstring. `run.py`, `model.py`, `verify.py` and `report.py` change only as listed in Section 3.2. CKPT records the diff. Tasks, data and mixes are regenerated from the task seed, and `run.py --check-only` and `verify.py` check them. CKPT also records that `tasks.json` and every `data/*.npz` file are byte-identical to Experiment 11's.

### 3.2 What changes

1. **Learner.** The gated learner is Experiment 11's LSTM arm, `config.json` `"arch": "lstm"`: token embedding 32, one LSTM layer with hidden size 128, and a 2-way linear head on the last hidden state. It reads the same `[task token, c_1, …, c_ℓ]` input over the same 34-token vocabulary and has **84,290 parameters**, which `run.py --check-only` asserts. An LSTM can simulate any DFA exactly, so the expressivity limit that stopped the transformer does not apply in principle.
2. **Optimizer unchanged and not retuned.** AdamW at learning rate $10^{-3}$, weight decay 0.01, betas (0.9, 0.999), batch 64, gradient-norm clip 1.0, and an 8-step linear warmup, as in Experiment 11 and its LSTM arm. There is no new learning-rate calibration. The arm already learns at $10^{-3}$, and a calibration on fresh seeds would add a tuning step that Experiment 11's gates did not have.
3. **Checkpoints.** All 8 checkpoints of every cell are saved, 4,096 × 84,290 × 4 bytes ≈ 1.38 GB, kept out of git with their SHA-256 in `ARTIFACT_SHA256.json`. The arm kept only $T = 2^{16}$.
4. **No pilot, calibration or separate LSTM arm.** `run.py --pilot`, `--calibrate-lr` and `--lstm` refuse to run. The exploratory arm serves as the disclosed pilot (Section 2.6).
5. **Wall cap and chunks (operational, not scientific).** The cumulative wall cap rises from 18,000 s to 28,800 s (8 h), because the CPU box is shared and Experiment 14 hit its cap under contention. The grid runs in resume-safe chunks of at most 2 h each, as many as needed. Every chunk is logged in `run_state.json`, `manifest.json` and CKPT. Done markers are written last, so no finished cell is ever retrained. This replaces Experiment 11's "resume once". If the cap is reached anyway, the analysis runs on the finished cells and RESULTS reports the missing ones. Cells run longest-level first in a fixed order.
6. **Manifest host.** `manifest.json` records a neutral host label (`cpu-box`) instead of the machine name.

### 3.3 Section 2.6 for this experiment: the $\varepsilon^\star$ rule (disclosed)

Experiment 11 fixed its $\varepsilon^\star$ levels with pre-written pilot rules P1, P1b and P2 (its Section 2.6). The acceptance for this rerun asks for the same gates. So the same rules are applied mechanically to the exploratory arm in place of a one-seed pilot. For each target, $T^\star$ is the geometric mean over the 8 seeds, using the same first-crossing log–log interpolation, with censored values at $2^{16}$. The rule is then applied to the median over the 16 targets.

| Rule | Condition | Arm value | Fires? |
|---|---|---|---|
| P1 | median $T^\star(0.1\ \text{bits}) \gt 2^{14}$ | 24,864 | yes: primary 0.3 bits |
| P1b | median $T^\star(0.3\ \text{bits}) \gt 2^{14}$ (16,384) | 16,947 | yes, by 3%: primary 0.5 bits |
| P2 | more than 4 of 16 targets at or below 0.1 bits at $T = 2^{10}$ (median over seeds) | 0 of 16 | no: window stays $2^9 … 2^{16}$ |

Per seed, the median $T^\star(0.3)$ ranged from 15,985 to 18,896, above $2^{14}$ in 6 of 8 seeds. The median $T^\star(0.5)$ over targets is 13,624.

**Consequences, fixed now.** The primary $\varepsilon^\star$ is **0.5 bits**, the levels are $\{0.5, 0.3, 0.1\}$, and G4's model-free fallback is 0.7 bits. This is exactly Experiment 11's frozen `config.json`, so every gate value and threshold is byte-identical to Experiment 11's. The decision to reuse Experiment 11's rules was made before these three medians were computed. It was not blind, though, because the arm's per-level $T^\star(0.1)$ table had been read. P1b fired by a small margin. Had it not, the primary level would have been 0.3 bits. RESULTS therefore reports every $T^\star$-based number at all three levels (0.5, 0.3 and 0.1 bits) with equal prominence. Only the 0.5-bit level is gated.

**P3 (time).** The arm's mean cell time under contention was 98.8 s, which projects 512 × 98.8 / 4 workers ≈ 12,600 s (3.5 h), below 90% of the cap. P3 does not fire.

## 4. Measurements, laws and analysis

Experiment 11's PROTOCOL Sections 3 and 4 apply unchanged: the per-checkpoint metrics, the per-string predictions, $T^\star$ with censoring flags, the plateau diagnostic, the resolution floor $\varepsilon_{\text{res}} = 10^{-3}$ bits with its censored residual, the six laws plus `M_K_floor`, the per-seed fits from 64 starts, the independent fitter, the two-stage analysis (forecasts hashed in CKPT before any held-out row is read), the seed-block and two-stage bootstraps, the multipliers, elasticities, $E_{\text{mf}}$ and range ratio, and the Section 4.6 mechanism analyses. Wherever that text says "0.1 bits (or the P1 level)", read 0.5 bits.

**Reported comparisons (no gate).**

- Experiment 20 next to Experiment 11, gate by gate: estimates, intervals and verdicts.
- Mean $\varepsilon(2^{16})$ by level and $\alpha$ for the LSTM and the transformer.
- The number of LSTM points below $\varepsilon_{\text{res}}$, by level. The LSTM drives $n = 4$ to about 0 bits, so many points will enter through the censored residual.

## 5. Prospective pass/fail gates

Experiment 11's PROTOCOL Section 5 applies unchanged: G1–G4 primary and G5–G7 secondary, with the same statistics and thresholds, at $\varepsilon^\star = 0.5$ bits where it applies. RESULTS labels each gate *prospective* or *not prospective* as Section 2 says.

**Predictions written before any $\alpha \lt 1$ LSTM run** (scored in RESULTS, not gates):

- G1 `SUPPORTED`: the arm already shows $T^\star$ rising steeply with $n$ (not prospective).
- G5 `NOT_ESTABLISHED`: at $\alpha = 1$, the $n = 4$ loss goes to about 0 while $n = 32$ stays high, so the gap should grow from $2^{12}$ to $2^{16}$ rather than shrink (not prospective).
- G2 `SUPPORTED`: loss now differs strongly with $K$ at every $\alpha$, so a $K$ law should forecast held-out cells well over 10% better than `M_align_eff`.
- G4 `NOT_ESTABLISHED`, with $E \lt 1$: if $n = 32$ stays mostly unlearned, the complexity elasticity is large, and per log unit complexity will cost more data than misalignment.
- G3: unresolved. The curves fall to the resolution floor rather than following a single power law, so neither sub-gate is clearly favoured.
- G6 `NOT_ESTABLISHED`, and $\eta$ below 1 (transfer, as for the transformer). Low confidence.

## 6. Artifacts and verification

As in Experiment 11's Section 8, under `expt_v1/results/`: `tasks.json`, `data/`, `mixes.json`, `metrics.csv` (4,096 rows), `predictions/`, `cells/`, `tstar.csv`, `fits.json`, `forecasts_frozen.json`, `analysis.json`, `manifest.json`, `verification.json` and `checkpoints/` (out of git, hashed).

`verify.py` must report PASS before anything is written up. Steps 1–9 are Experiment 11's: source hashes, task and covariate regeneration with Hopcroft and Moore, data and mix regeneration, a replay of every checkpoint with the LSTM (`max_abs_diff < 1e-9` bits), $T^\star$, both fitters, and every gate recomputed. **Step 10 is reported but not part of PASS.** It compares the 1,024 $\alpha = 1$ metric rows with Experiment 11's exploratory arm and gives the maximum absolute difference, which is expected to be 0.

## 7. Scope and limits

- Experiment 11's Section 9 limits all hold: $K$ is exact for $\mathcal L_{\text{DFA}}$ and this sampler only. There is one model size, one learning rate and lengths 8–24, and the distractors are unrelated languages.
- This is a different learner, not a fix of the transformer result. LSTMs simulate DFAs natively, so complexity may cost an LSTM less than it costs a transformer (or a language model). A verdict here holds for this recurrent learner.
- The $\alpha = 1$ slice was seen (Section 2). The prospective evidence is the $\alpha \lt 1$ cells and the held-out forecasts.
- The laws are power laws in $T$. An LSTM curve that drops to the floor within the window is not a power law, and the censored residual only partly handles it. RESULTS reports the fraction of points below the floor.
- $n = 32$ may still be mostly unlearned at $2^{16}$ (31 of 32 arm runs were censored at 0.1 bits), so $T^\star$ at $n = 32$ stays capped and $M_K$ is a lower bound.

**TLDR-end:** [tdc: Expt 20 design] Pre-registered rerun of Experiment 11's 512-run DFA grid (same 16 exact-size target languages, target shares, 8 seeds, checkpoints, analysis and gates) with the 84,290-parameter LSTM that learned these languages in Experiment 11's exploratory arm. The α = 1 slice was already seen, so G1, G5 and G7 are not prospective, and the test rests on the 384 unseen α < 1 cells and frozen held-out forecasts.
