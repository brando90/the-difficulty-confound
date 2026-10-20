"""Apply Experiment 11's pilot rules P1, P1b and P2 to Experiment 11's exploratory LSTM arm (PROTOCOL Section 3.3).

  eps_rule_from_exp11_arm.py      writes eps_rule.json
Per target: T* = geometric mean over the 8 seeds (first crossing, log-log interpolation, censored at 2^16);
the rules use the median over the 16 targets.
"""
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import t_star  # noqa: E402

c = json.loads((HERE / 'config.json').read_text())
rows = list(csv.DictReader(open(HERE / c['exp11_lstm_arm_metrics'])))
Ts = c['T_checkpoints']
eps = collections.defaultdict(dict)
for r in rows:
    eps[(r['target_id'], int(r['seed']))][int(r['T'])] = float(r['excess_bits'])
targets = sorted({k[0] for k in eps})
seeds = sorted({k[1] for k in eps})
out = dict(source=c['exp11_lstm_arm_metrics'], n_rows=len(rows), per_target_gm_tstar={}, median={}, per_seed_median={})
for es in (0.1, 0.3, 0.5):
    gm = {t: float(np.exp(np.mean([np.log(t_star(Ts, [eps[(t, s)][T] for T in Ts], es)[0]) for s in seeds])))
          for t in targets}
    out['per_target_gm_tstar'][str(es)] = gm
    out['median'][str(es)] = float(np.median(list(gm.values())))
    out['per_seed_median'][str(es)] = {s: float(np.median([t_star(Ts, [eps[(t, s)][T] for T in Ts], es)[0]
                                                           for t in targets])) for s in seeds}
out['P2_targets_at_or_below_0p1_at_T1024'] = int(sum(np.median([eps[(t, s)][1024] for s in seeds]) <= 0.1
                                                     for t in targets))
out['P1_fires'] = out['median']['0.1'] > 2 ** 14
out['P1b_fires'] = out['median']['0.3'] > 2 ** 14
out['P2_fires'] = out['P2_targets_at_or_below_0p1_at_T1024'] > 4
out['eps_star_primary'] = 0.5 if out['P1b_fires'] else 0.3 if out['P1_fires'] else 0.1
assert out['eps_star_primary'] == c['eps_star_primary']
(HERE / 'eps_rule.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('median', 'P1_fires', 'P1b_fires', 'P2_fires', 'eps_star_primary')}, indent=1))
