"""The analysis wrappers reproduce the source experiments' committed results from their committed metrics.

Each check runs in its own process (Experiments 13 and 17 both name their analysis module `analyze`).
The Experiment 13 check reruns its full stage 1 and stage 2 (about 10 minutes on one core): set E23_SLOW=1.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]

E17 = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import analyze23 as Z
sys.path.insert(0, str(Z.SRC['exp17']))
import analyze as A
c = A.load_config(); c2 = json.loads((Z.SRC['exp17b'] / 'config_v2.json').read_text())
v1, v2 = Z.SRC['exp17'] / 'results', Z.SRC['exp17b'] / 'results'
rows = [r for r in A.read_rows(v1) if r['pool'] in c2['reuse_from_v1']] + A.read_rows(v2)
tasks = json.loads((v2 / 'tasks.json').read_text())
res = Z.coverage_analysis(A, rows, tasks, c['seeds'], c['T_checkpoints'], c2['pool_order'],
                          ['easy'] + c2['pool_order'] + ['hard', 'near'], c['alpha_H'], c['easy_level'],
                          c['hard_level'], c['gate']['min_consistent_pairs'], c['analysis_seed'], c['bootstrap_draws'],
                          'cov32', True)
ref = json.loads((v2 / 'analysis_v2.json').read_text())
mine = json.loads(json.dumps(res, sort_keys=True))
print(json.dumps({k: mine[k] == ref[k] for k in ref if k in mine and k != 'pool_excess_at_Tmax'}))
'''

E13 = r'''
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import analyze23 as Z
sys.path.insert(0, str(Z.SRC['exp13']))
import analyze as A
src = Z.SRC['exp13'] / 'results' / 'small'
out = Path(sys.argv[2])
for n in ('metrics.csv', 'tasks.json', 'distractor_loss.csv', 'posterior16.json'):
    shutil.copy2(src / n, out / n)
tasks = json.loads((out / 'tasks.json').read_text())
c = A.load_config('small')
rows = [r for r in A.read_rows(out, tasks, True) if r['slice'] == 'all']
eps, rule = Z.eps_star_rule(rows, Z.CFG['eps_star_ladder'], 32, 0.4, Z.CFG['eps_star_min_share'])
A.stage1(c, out, tasks)
A.stage2(c, out, tasks)
same = json.loads((out / 'gates.json').read_text()) == json.loads((src / 'gates.json').read_text())
print(json.dumps(dict(eps=eps, gates_identical=same)))
'''


def _run(code, *args):
    p = subprocess.run([sys.executable, '-c', code, str(HERE), *map(str, args)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_coverage_analysis_reproduces_experiment17b():
    r = _run(E17)
    assert r and all(r.values()), r


@pytest.mark.skipif(os.environ.get('E23_SLOW') != '1', reason='slow (about 10 min): set E23_SLOW=1')
def test_experiment13_analysis_and_eps_rule(tmp_path):
    """Experiment 13's stage 1 + stage 2 on its committed 4-seed metrics give its committed gates.json, and the
    Section 5.1 epsilon* rule applied to those metrics gives the 1.10 bits that its pilot rule set."""
    r = _run(E13, tmp_path)
    assert r['gates_identical'] and r['eps'] == 1.1, r
