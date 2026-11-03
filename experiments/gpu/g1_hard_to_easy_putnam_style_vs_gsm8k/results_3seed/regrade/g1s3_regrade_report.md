# g1 seeds3 report (16/16 runs DONE, 1.924 GPU-hours of wall time)

| run | GSM8K | MATH-500 | L4-5 | stopped % | boxed % | mean tokens | L1 bpb | L3 bpb | HellaSwag |
|---|---|---|---|---|---|---|---|---|---|
| L1_gradeschool1__T152x32x2048__lr5e-06__s0 | 42.08 | 27.2 | 12.6 | None | None | None | 0.1344 | 0.3793 | 49.55 |
| L1_gradeschool1__T152x32x2048__lr5e-06__s1 | 42.53 | 26.6 | 12.21 | None | None | None | 0.1344 | 0.3793 | 49.2 |
| L1_gradeschool1__T152x32x2048__lr5e-06__s2 | 43.97 | 26.4 | 12.98 | None | None | None | 0.1344 | 0.3795 | 49.45 |
| L1_gradeschool1__T152x32x2048__lr2e-05__s0 | 43.44 | 22.0 | 10.31 | None | None | None | 0.1365 | 0.3981 | 48.65 |
| L1_gradeschool1__T152x32x2048__lr2e-05__s1 | 40.71 | 21.6 | 9.16 | None | None | None | 0.137 | 0.3994 | 49.25 |
| L1_gradeschool1__T152x32x2048__lr2e-05__s2 | 38.89 | 22.6 | 6.87 | None | None | None | 0.1372 | 0.3995 | 50.05 |
| L3_olympiad1__T152x32x2048__lr5e-06__s0 | 44.12 | 26.2 | 12.98 | None | None | None | 0.2126 | 0.3468 | 49.0 |
| L3_olympiad1__T152x32x2048__lr5e-06__s1 | 43.44 | 27.2 | 15.65 | None | None | None | 0.2125 | 0.3469 | 48.95 |
| L3_olympiad1__T152x32x2048__lr5e-06__s2 | 43.82 | 26.8 | 13.36 | None | None | None | 0.212 | 0.3469 | 49.0 |
| L3_olympiad1__T152x32x2048__lr2e-05__s0 | 38.59 | 20.6 | 9.16 | None | None | None | 0.2275 | 0.3507 | 47.8 |
| L3_olympiad1__T152x32x2048__lr2e-05__s1 | 38.51 | 24.4 | 10.69 | None | None | None | 0.2331 | 0.351 | 49.7 |
| L3_olympiad1__T152x32x2048__lr2e-05__s2 | 40.33 | 20.6 | 9.54 | None | None | None | 0.2316 | 0.3512 | 48.25 |
| L1_gradeschool0.5+L3_olympiad0.5[answer_only]__T152x32x2048__lr5e-06__s0 | 8.64 | 8.6 | 4.96 | None | None | None | 0.238 | 0.3777 | 48.85 |
| L1_gradeschool0.5+L3_olympiad0.5[answer_only]__T152x32x2048__lr5e-06__s1 | 8.34 | 7.4 | 3.05 | None | None | None | 0.237 | 0.3773 | 49.3 |
| L1_gradeschool0.5+L3_olympiad0.5[answer_only]__T152x32x2048__lr5e-06__s2 | 8.72 | 7.8 | 3.82 | None | None | None | 0.24 | 0.3789 | 49.3 |
| base | 28.73 | 20.0 | 8.4 | None | None | None | 0.1948 | 0.3659 | 48.7 |

## Readout (points; per seed, mean, 95% CI over items; t over seeds)

- gsm8k_E_minus_H@5e-06: [-2.05, -0.91, 0.15] mean -0.94 CI [-2.98, 1.09] t [-3.67, 1.8]
- math500L45_H_minus_E@5e-06: [0.38, 3.44, 0.38] mean 1.4 CI [-1.78, 4.58] t [-2.98, 5.78]
- gsm8k_E_minus_H@2e-05: [4.85, 2.2, -1.44] mean 1.87 CI [0.03, 3.69] t [-5.98, 9.72]
- math500L45_H_minus_E@2e-05: [-1.15, 1.53, 2.67] mean 1.02 CI [-1.91, 4.07] t [-3.85, 5.88]
- gsm8k_E@5e-06_minus_F: [33.43, 34.19, 35.25] mean 34.29 CI [31.72, 36.85] t [32.02, 36.56]
- math500_L45_E@5e-06_minus_F: [7.63, 9.16, 9.16] mean 8.65 CI [4.83, 12.34] t [6.46, 10.84]
- gsm8k_E@5e-06_minus_base: [13.34, 13.8, 15.24] mean 14.13 CI [11.68, 16.55] t [11.67, 16.58]
- gsm8k_E@2e-05_minus_F: [34.8, 32.37, 30.17] mean 32.45 CI [29.87, 34.93] t [26.7, 38.2]
- math500_L45_E@2e-05_minus_F: [5.34, 6.11, 3.05] mean 4.83 CI [1.65, 7.89] t [0.89, 8.78]
- gsm8k_E@2e-05_minus_base: [14.71, 11.98, 10.16] mean 12.28 CI [9.91, 14.63] t [6.59, 17.97]
- gsm8k_H@5e-06_minus_F: [35.48, 35.1, 35.1] mean 35.23 CI [32.55, 37.86] t [34.69, 35.77]
- math500_L45_H@5e-06_minus_F: [8.02, 12.6, 9.54] mean 10.05 CI [5.98, 14.12] t [4.26, 15.84]
- gsm8k_H@5e-06_minus_base: [15.39, 14.71, 15.09] mean 15.06 CI [12.56, 17.56] t [14.21, 15.91]
- gsm8k_H@2e-05_minus_F: [29.95, 30.17, 31.61] mean 30.58 CI [28.13, 32.95] t [28.33, 32.83]
- math500_L45_H@2e-05_minus_F: [4.2, 7.63, 5.73] mean 5.85 CI [2.29, 9.41] t [1.58, 10.13]
- gsm8k_H@2e-05_minus_base: [9.86, 9.78, 11.6] mean 10.41 CI [8.04, 12.81] t [7.85, 12.97]
- gsm8k_F_minus_base: [-20.09, -20.39, -20.02] mean -20.17 CI [-22.87, -17.46] t [-20.67, -19.67]
- math500L45_F_minus_base: [-3.44, -5.34, -4.58] mean -4.45 CI [-8.02, -1.02] t [-6.84, -2.07]
- gsm8k_interaction_lr:  mean 2.81 CI [0.66, 5.03]

## Verdicts

- sign_flip: LR-DEPENDENT, SIGN NOT ESTABLISHED
- P1_E_minus_H_ge_3_at_5e-6: NOT SUPPORTED
- H_beats_format_only_on_gsm8k_at_5e-6: YES
- format_share_of_H_gain: -1.339
- F_answers_without_working: False
