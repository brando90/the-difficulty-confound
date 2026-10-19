"""Pilot 2 (PROTOCOL Section 6.1, disclosed): pilot 1 showed the mod-10 L4 targets and the hard pool still at chance
at 2^17 pairs. Runs a few pilot-seed runs to 2^20 to see whether the plateau ends with more data, and whether the
same-K additive targets (a+b, a-b) learn. Output: results/pilot2/. Usage: python pilot2.py --pool 3 [--extra JSON]"""
import argparse
import json
import multiprocessing as mp
import os
import time

import run

T20 = [2 ** k for k in range(10, 21)]
SPECS = [
    dict(arm='pilot', dataset='mnist', mismatch='rot90', target='sum10', alpha=1.0, seed=18909),
    dict(arm='pilot', dataset='mnist', mismatch='rot90', target='sum19', alpha=1.0, seed=18909),
    dict(arm='H', dataset='mnist', mode='pool_only', pool='hard', seed=18909),
]


def one(spec):
    run.CFG['T_checkpoints'][spec['dataset']] = T20
    run.OUT = os.path.join(run.HERE, 'results', 'pilot2')
    return run.run_one(spec)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=3)
    ap.add_argument('--extra', default='[]')
    a = ap.parse_args()
    specs = SPECS + json.loads(a.extra)
    with mp.get_context('fork').Pool(a.pool) as P:
        for r, st in P.imap_unordered(one, specs):
            print(time.strftime('%H:%M:%S'), r, st, flush=True)
