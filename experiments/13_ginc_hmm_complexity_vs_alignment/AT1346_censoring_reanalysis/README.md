# AT-1346: censoring-aware re-analysis of Exp 13-GPU (Laolu's 360 saved runs), CPU only

## Result
- With censored runs kept, pooled E is about 1: 1.09 [1.06, 1.12] at target loss eps* 0.02, 1.03 [0.99, 1.07] at 0.05, 0.98 [0.93, 1.03] at 0.10. Crossing times do not support the 0.575 that Laolu's law (fixed before the runs) gives.
- The alpha x K interaction law beats the separable law by 4.6% [3.8, 5.3] in held-out error when one target is left out. It gives no gain at the held-out complexity level n=32 (RMSE 0.829 vs 0.823).
- The claim that E is established (the protocol's gate G4) stays not established.

## Data
`gpu_all_slice.csv` = token-weighted "all" slice of Laolu's saved `metrics.csv` (SNAP, Exp 13-GPU, expt_v1/results/gpu). 3 sizes (3.2M/25.4M/85.3M) x 10 targets (2 per complexity n = 8/16/32/64/128 hidden states) x 4 alignment shares alpha (0.05, 0.15, 0.4, 1) x 3 seeds = 360 runs, 5 checkpoints each (T = 4.2M ... 1.07B tokens; L stops at 0.54B) = 1,800 (run, checkpoint) rows. Outcome: learning term = excess bits minus the exact identification floor. Nothing launched, nothing paid, Laolu not contacted. Scripts: `analyze.py` (part A), `lawB.py` (part B); outputs `partA_E.json`, `partB_law.json`.

## A. E = e_alpha / e_K per target loss, censoring handled
Per run, T* = tokens to reach the learning-term target eps*, log-log interpolated between checkpoints. A run that never gets there is **right-censored** (T* >= last checkpoint), one already there at the first checkpoint is left-censored (none occurred). Tobit regression of ln T* on ln alpha, ln n (size fixed effects when pooled), so censored runs stay in with their censored likelihood instead of being dropped or counted at the last checkpoint. e_alpha = -d ln T*/d ln alpha, e_K = d ln T*/d ln n, E = e_alpha/e_K. Denominators: 120 runs per size per eps*, 360 pooled.

Right-censored runs / 120 per size (S, M, L; pooled /360):
- eps* 0.02: 71, 59, 54 (184/360 = 51%)
- eps* 0.05: 25, 15, 7 (47/360 = 13%)
- eps* 0.10: 6, 0, 0 (6/360 = 2%)

E (point [95% target-level bootstrap]), K = n states; pooled = size fixed effects:
| eps* | S | M | L | pooled | pooled, K in bits |
|---|---|---|---|---|---|
| 0.02 | 1.11 [1.08, 1.14] | 1.08 [1.01, 1.16] | 1.09 [0.99, 1.20] | 1.09 [1.06, 1.12] | 1.15 [1.12, 1.18] |
| 0.05 | 1.04 [1.01, 1.07] | 1.03 [0.95, 1.11] | 1.01 [0.93, 1.08] | 1.03 [0.99, 1.07] | 1.09 [1.04, 1.13] |
| 0.10 | 0.98 [0.95, 1.00] | 1.02 [0.95, 1.09] | 0.91 [0.80, 1.03] | 0.98 [0.93, 1.03] | 1.03 [0.97, 1.09] |

Pooled e_alpha / e_K: eps* 0.02: 1.04 / 0.95; 0.05: 0.68 / 0.66; 0.10: 0.44 / 0.45 (every e_K CI is above 0, in no bootstrap draw was e_K <= 0). With an alpha x n interaction term in the T* regression, the alpha-elasticity grows with complexity (interaction coefficient -0.27 [-0.31, -0.23] pooled at eps* 0.02, similar at 0.05 and 0.10), and E at the centre point stays 1.05 / 1.02 / 0.98.

Reading: once censored runs are kept, **E is about 1 at every eps* (0.98 to 1.09 pooled), not 0.58**. Alignment and complexity cost about the same data at the chosen targets; neither dial clearly dominates. It is a censoring-aware version of the earlier model-free estimate, not the frozen law's E (below).

## B. Interaction law vs separable law, held out
Separable law (the shape of Laolu's M_K_coef law without its identification-floor offset, since the learning term already has the floor removed): ln L = c + kappa ln n + b ln alpha - gamma ln T (size intercepts when pooled). Interaction law adds delta ln n ln alpha (alignment cost depends on complexity). Also a variant with a complexity-dependent learning rate (adds gamma1 ln n ln T, columns 'rate') and one with both extras (columns 'both'). Fit by OLS on ln(learning term) over runs/checkpoints whose learning term is above the 0.015-bit resolution floor (1,571 of 1,800 rows kept; 229 dropped = 12.7%); "clip" variant keeps all 1,800 with values clipped at 0.015 (below).

Held-out RMSE in log2 units (factor 2^RMSE), pooled, rows above the floor:
| held out | separable | rate variant | **interaction** | both |
|---|---|---|---|---|
| one target at a time (10 folds) | 0.904 | 0.898 | **0.862** | 0.860 |
| one complexity level at a time (5 folds) | 0.935 | 0.926 | **0.895** | 0.891 |
| of which level n=32 (the level the original protocol held out) | 0.823 | 0.821 | 0.829 | 0.827 |
| of which n=128 / n=8 | 0.869 / 1.269 | 0.858 / 1.249 | 0.857 / 1.151 | 0.848 / 1.144 |
| one alpha at a time (4 folds) | 0.975 | 0.988 | **0.940** | 0.956 |

Relative RMSE reduction of the interaction law over separable (point [95% target-level bootstrap of the held-out errors]): target 4.6% [3.8, 5.3]; level 4.2% [3.5, 4.9]; alpha 3.5% [2.9, 4.1]. Per size, target-held-out: S 2.2% [1.5, 2.9], M 4.9% [3.9, 5.8], L 14.0% [12.1, 15.6]; level-held-out: S 0.8% [0.1, 1.5], M 4.5% [3.5, 5.3], L 14.6% [13.1, 16.1]. Clip variant, pooled: 5.3% / 6.1% / 6.1% (target / level / alpha), 1,800 rows.
So the interaction law fits held-out data a little better (about 4-5% pooled, 12-15% at L, about 1% at S for a held-out level), and it does not beat the separable law at the held-out level n=32 (0.829 vs 0.823). Neither law is good in absolute terms: held-out error is about 0.9 log2 units, a factor of 1.9 on the learning term.

Coefficients (refit on all rows, 95% target bootstrap, 1,000 resamples), pooled: separable kappa 0.393 [0.374, 0.411], b -0.333 [-0.347, -0.319], gamma 0.828; interaction + rate variant delta -0.156 [-0.172, -0.141], gamma1 -0.020 (x*t coefficient +0.020 [0.015, 0.025]). delta is negative at every size (S -0.100, M -0.145, L -0.213; all CIs exclude 0).

Law-implied E (elasticity ratio at the centre of the grid), pooled:
- separable: 0.85 [0.82, 0.88], the same at every eps* (e_alpha 0.40, e_K 0.47); per size S 0.87, M 0.86, L 0.77.
- interaction + rate variant: eps* 0.02: 0.77 [0.74, 0.79]; 0.05: 0.81 [0.78, 0.84]; 0.10: 0.85 [0.82, 0.88]; per size at 0.02: S 0.91, M 0.69, L 0.60.
Laolu's frozen value (his M_K_both law, with an identification-floor offset and a complexity-dependent exponent): 0.575 [0.567, 0.586]. My laws, which have no offset term and fit the learning term in log space, give 0.77-0.85.

## What the two parts say together
- The law-based E is below 1 (0.58 frozen; 0.77-0.85 here); the T*-based E with censoring kept is about 1 (0.98-1.09). They differ because a power law fit over all five checkpoints is dominated by early, high-loss rows and by the plateau near the resolution floor, while T* at a fixed eps* looks only at the crossing. So E < 1 is a property of the law fit, not of the crossing times.
- Interaction (alignment cost rises with complexity) is real in every size and every eps* (T* regression delta -0.15 to -0.47; law delta -0.10 to -0.21), and it is what a separable law cannot represent; it buys 4-5% held-out error pooled, more at L.
- E is not established as different from 1. Its interval straddles 1 at eps* 0.05 [0.99, 1.07] and 0.10 [0.93, 1.03]. At 0.02 it is 1.09 [1.06, 1.12], slightly above 1, but that interval is the optimistic one (limits 1-2).

## Limits (all disclosed)
1. Only 10 targets (2 per complexity level). The bootstrap resamples targets within level (all sizes, seeds, alphas travel together); with 2 per level it understates uncertainty, so read the intervals as a lower bound. Bootstrap of held-out errors does not refit.
2. Five checkpoints spaced 4x apart; T* is a log-log interpolation. At eps* 0.02, 51% of runs are censored (S 59%, S's n=128 runs all censored), so those e values lean on the Tobit's normal-error extrapolation.
3. Single Tobit sigma; heteroskedasticity by size not modelled (per-size fits shown).
4. My laws omit the identification-floor offset and complexity-dependent exponent of Laolu's frozen law; I did not refit his exact nonlinear laws or reproduce his forecasts.csv. Results are exploratory, not pre-registered gates.
5. Data are Laolu's saved metrics only (on SNAP, copied earlier); not independently regenerated.

TLDR-end: [Exp 13-GPU: censoring re-analysis] E≈1 with censoring kept; interaction law +4–5% held out; G4 (E established) still not established.
