# Addition exposure results

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/01_fixed_width_addition_exposure/expt_v1/results/table.md>

**TLDR:** NOT_ESTABLISHED_IN_THIS_SETUP. All numbers below are from the fixed test set.

| Training condition | Maximum carry chain | Exact sums / predictions | Accuracy, 95% seed interval | Conditional bits / answer, 95% seed interval |
|---|---:|---:|---:|---:|
| scarce_long_carry | 0 | 8000/8000 | 100.00% [100.00, 100.00] | 0.021 [0.018, 0.024] |
| scarce_long_carry | 1 | 710/8000 | 8.88% [8.08, 9.65] | 10.241 [9.446, 11.048] |
| scarce_long_carry | 2 | 695/8000 | 8.69% [7.50, 9.84] | 7.940 [7.454, 8.443] |
| scarce_long_carry | 3 | 768/8000 | 9.60% [7.85, 11.22] | 6.564 [6.104, 6.997] |
| balanced | 0 | 7514/8000 | 93.92% [91.61, 95.89] | 0.540 [0.438, 0.681] |
| balanced | 1 | 5914/8000 | 73.92% [70.83, 77.51] | 0.958 [0.853, 1.088] |
| balanced | 2 | 6916/8000 | 86.45% [84.15, 88.50] | 0.679 [0.591, 0.799] |
| balanced | 3 | 7591/8000 | 94.89% [93.45, 96.25] | 0.475 [0.405, 0.574] |
| rich_long_carry | 0 | 547/8000 | 6.84% [5.79, 7.79] | 7.538 [7.171, 7.972] |
| rich_long_carry | 1 | 315/8000 | 3.94% [2.86, 5.22] | 11.203 [10.277, 11.998] |
| rich_long_carry | 2 | 1481/8000 | 18.51% [16.49, 20.72] | 7.118 [6.595, 7.568] |
| rich_long_carry | 3 | 8000/8000 | 100.00% [100.00, 100.00] | 0.017 [0.016, 0.018] |
| shuffled_labels | 0 | 5/8000 | 0.06% [0.03, 0.09] | 11.793 [11.676, 11.908] |
| shuffled_labels | 1 | 5/8000 | 0.06% [0.01, 0.11] | 11.639 [11.590, 11.680] |
| shuffled_labels | 2 | 5/8000 | 0.06% [0.03, 0.10] | 11.507 [11.425, 11.580] |
| shuffled_labels | 3 | 7/8000 | 0.09% [0.05, 0.12] | 11.434 [11.312, 11.573] |

Recovered fraction of baseline gap: {'mean': 0.9434457964601771, 'lo': 0.9273867595818817, 'hi': 0.9585661714597247}
Half-gap sign-flip p-value: 0.00390625

Paired percentile bootstrap over eight training/data-sampling seeds; 20000 draws. Conditional on this fixed test set, architecture, split and optimizer.

Accuracy and code-length table entries are descriptive: p-val=n/a. Code lengths exclude the model and training data and are not Kolmogorov complexity or a complete minimum-description-length code.

## Prespecified criteria

- baseline_gap_at_least_20_percentage_points: True
- baseline_no_carry_accuracy_at_least_90_percent: True
- balanced_long_carry_accuracy_at_least_90_percent: True
- no_carry_accuracy_drop_at_most_5_percentage_points: False
- recovery_95_percent_lower_bound_above_half: True
- shuffled_label_overall_accuracy_below_5_percent: True
