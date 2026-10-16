# Public artifact dictionary

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/03_answer_matched_alignment/DATA_DICTIONARY.md>

**TLDR:** The archived run contains the observations, forecasts, training IDs and weights needed to reproduce every reported value. Compressed NumPy files can be read with `numpy.load`; they are not model-generated summaries.

All paths below are under `expt_v1/results/`.

| File pattern | Contents |
|---|---|
| `heldout_items.npz` | Full original unordered population arrays `a`, `b`, `carry`; selected `answers`; fixed `test`, `calibration`, `probe` item-ID arrays. The integer answer is `a[item_id] + b[item_id]`. |
| `train_<seed>.npz` | Five `q0.10` through `q0.90` arrays of intervention training IDs. `q0.50` is also the common warmup pool. IDs and swaps never overlap held-out splits. |
| `<split>_<seed>_<width>_<q>_<step>.npz` | Aligned arrays `item_id`, `prediction` (integer answer), `bits` (sum of four digit negative log probabilities, base two). There are 1,200 test/calibration files and 1,684,800 item observations across all checkpoints. |
| `probe_<seed>_<width>.npz` | Same schema, from the 40 warm models before intervention; 37,440 probe observations. |
| `test_metrics.csv`, `calibration_metrics.csv` | Seed, width, mixture, updates, carry chain; parameter count, intervention-stratum fraction, independent starting bits; mean bits, number correct and denominator. Each table contains 1,200 aggregate rows. |
| `probe_features.csv` | Eighty mean starting-difficulty observations: eight seeds × five widths × two strata, each averaging 468 probe items. |
| `model_*.pt`, `warm_*.pt` | 200 final and 40 warm PyTorch parameter dictionaries. Load only with `torch.load(..., weights_only=True)`. |
| `answer_counts.csv` | Population and split counts for every selected answer and carry group. |
| `input_digit_marginals.csv` | Training-input digit frequencies by seed, mixture and each of the six input positions. These marginals are not controlled to equality. |
| `warmup_union_counts.csv` | Verified intervention, warmup and union sizes per seed and mixture. |
| `frozen_forecasts.json` | Calibration-only coefficients, fit-row IDs and input/source hashes. |
| `forecasts.csv`, `scored_forecasts.csv` | All 3,840 held-out predictions across five laws, with observed bits and signed errors added only in the scored table. |
| `prediction_metrics.csv`, `prediction_comparisons.csv` | Every law and held-out split, errors, bias, both means, paired-seed uncertainty and all declared feature contrasts. |
| `intervention_seed_width.csv` | Forty paired primary-contrast rows, retaining seed/width dependence. |
| `accuracy_summary.csv` | All 150 checkpoint × width × mixture × carry means, seed standard deviations and pointwise intervals. |
| `runtime.json`, `cells.json`, `job_status.json`, `verification.json` | Actual execution identity, source/config hashes, full 200-cell accounting, process exit state and deterministic verification evidence. |

The independent uncertainty unit is a complete paired seed block (eight blocks), not an item, width or checkpoint. Reusing the fixed test set controls within-study comparisons but does not measure uncertainty over fresh test populations.

Minimal observation read:

```python
from pathlib import Path
import numpy as np
r = Path('expt_v1/results')
items = np.load(r / 'heldout_items.npz')
z = np.load(r / 'test_2101_32_0.10_3000.npz')
ids = z['item_id']
target = items['a'][ids] + items['b'][ids]
correct = z['prediction'] == target
print(correct.mean(), z['bits'].mean())
```

`verify.py` recomputes aggregate rows, independently reconstructs every forecast and paired-bootstrap comparison, and replays all 240 saved parameter dictionaries. It never retrains a model or changes a scientific result.
