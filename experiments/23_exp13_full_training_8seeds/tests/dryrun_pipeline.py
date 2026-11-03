"""Pipeline dry run: the whole chain (init, stage, decide, aggregate, analyze23, verify23, report23) on a miniature
copy of the experiment: 2 seeds per experiment, stages 2^13 and 2^14 tokens, 32 eval documents. Numbers are
meaningless; it checks that every step runs and every file the next step needs is written.

Usage: python tests/dryrun_pipeline.py --work DIR [--workers 4]      (about 20-40 min on 4 cores)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def sh(cmd, cwd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    shutil.rmtree(work, ignore_errors=True)
    exps = work / 'experiments'
    exps.mkdir(parents=True)
    for p in HERE.parent.iterdir():  # the source experiments, linked
        if p.name != HERE.name and p.is_dir():
            os.symlink(p, exps / p.name)
    code = exps / HERE.name
    shutil.copytree(HERE, code, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'), symlinks=True)
    cfg = json.loads((code / 'config.json').read_text())
    cfg['e13']['seeds'] = cfg['e13']['seeds'][:2]
    cfg['e17b']['seeds'] = cfg['e17b']['seeds'][:2]
    cfg['e17b']['min_consistent_pairs'] = 4
    cfg['stages_T'] = [8192, 16384]
    cfg["T_checkpoints"] = [2048, 4096, 8192, 16384]
    cfg['stop_rule'] = dict(window_doublings=2, abs_bits=0.0, rel=0.0)
    (code / 'config.json').write_text(json.dumps(cfg, indent=1))
    out = work / 'out'
    py = sys.executable
    sys.path.insert(0, str(code / 'tests'))
    from test_run23 import _shrink  # noqa: E402
    for exp in ('e13', 'e17b'):
        sh([py, 'run23.py', 'init', '--exp', exp, '--output', out / exp], code)
        _shrink(out / exp, n_eval=16, n_val=8)
        n = int(subprocess.run([py, 'run23.py', 'count', '--exp', exp], cwd=code, capture_output=True, text=True,
                               check=True).stdout.split()[1])
        for k in range(2):
            for i in range(n):
                sh([py, 'run23.py', 'stage', '--exp', exp, '--stage', k, '--task', i, '--output', out / exp,
                    '--workers', a.workers], code)
            sh([py, 'run23.py', 'decide', '--exp', exp, '--stage', k, '--output', out / exp], code)
        sh([py, 'run23.py', 'aggregate', '--exp', exp, '--output', out / exp], code)
    sh([py, 'analyze23.py', '--exp', 'e17b', '--output', out / 'e17b'], code)
    sh([py, 'analyze23.py', '--exp', 'e13', '--output', out / 'e13'], code)
    p = subprocess.run([py, 'verify23.py', '--out', out, '--quick'], cwd=code)
    sh([py, 'report23.py', '--out', out, '--results', work / 'RESULTS_dryrun.md'], code)
    print(f'DRY RUN DONE (verify exit {p.returncode}); report: {work / "RESULTS_dryrun.md"}')


if __name__ == '__main__':
    main()
