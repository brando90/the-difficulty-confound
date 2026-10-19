"""Experiment 19 verification (PROTOCOL Section 6). Must print PASS before any number is reported.

  python verify.py --results results [--replay all|N]

1. Training sources (Experiment 19 and the Experiment 11 files it imports) match manifest.json.
2. Every metrics.csv row is recomputed from predictions/T_<T>.npz: excess_bits and per-length bits exactly (same
   float32 values, float64 means), accuracy as the share of strings under 1 bit (ties within 1/4096), and
   target/distractor counts from round(alpha * T).
3. The final checkpoint of every kept cell (or N of them) is loaded into Experiment 11's model.py on the CPU and
   evaluated on the regenerated test set; per-string bits must match the GPU predictions (mean within 1e-4 bits).
4. The test set and the cell labels are regenerated from Experiment 11's code and tasks.
5. analyze.py stages 1 and 2 and summary.py are rerun in a scratch copy; fits, frozen forecasts, every gate, E at
   every level and the summary must be identical.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import common as C

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--replay', default='all')
    ap.add_argument('--skip-analysis', action='store_true', help='smoke tests only; a skipped check is never a PASS')
    args = ap.parse_args()
    res = Path(args.results)
    man = json.loads((res / 'manifest.json').read_text())
    st = json.loads((res / 'run_state.json').read_text())
    V = {}
    V['1_sources_match_manifest'] = all(sha(HERE / f) == h for f, h in man['sources'].items()) and \
        all(sha(C.E11 / f) == h for f, h in man['e11_sources'].items())

    c11 = C.load_c11()
    tasks = {t['task_id']: t for t in C.load_tasks()}
    tb, tl = C.D11.build_test_set(c11)
    ref = np.load(C.E11 / 'results' / 'data' / 'dfa_n04_t1_test.npz')
    V['4_test_set_identical_to_exp11'] = bool(np.array_equal(ref['bits'].astype(np.uint64), tb.astype(np.uint64))
                                              and np.array_equal(ref['length'], tl))
    Ls = np.unique(tl)
    rows = list(csv.DictReader(open(res / 'metrics.csv')))
    by = {(r['seed'], r['target_id'], f"{float(r['alpha']):.2f}", int(r['T'])): r for r in rows}
    n_ok = n_rows = 0
    acc_max = 0.0
    for T in st['T_done']:
        z = np.load(res / 'predictions' / f'T_{T}.npz')
        bits, cells = z['bits'], z['cells']
        for i, cell in enumerate(cells):
            s, rest = str(cell).split('_', 1)
            tid, a = rest.rsplit('_', 1)
            r = by[(s, tid, a, T)]
            pr = bits[i].astype(np.float64)
            bl = [float(pr[tl == L].mean()) for L in Ls]
            tseen = int(np.floor(float(a) * T + 0.5))
            ok = (float(r['excess_bits']) == float(pr.mean()) and json.loads(r['excess_bits_by_length']) == bl
                  and int(r['target_examples_seen']) == tseen and int(r['distractor_examples_seen']) == T - tseen)
            acc_max = max(acc_max, abs(float(r['accuracy']) - float((bits[i] < 1.0).mean())))
            n_ok += ok
            n_rows += 1
    V['2_metrics_rows_recomputed'] = f'{n_ok}/{n_rows}'
    V['2_rows_equal_csv'] = n_rows == len(rows)
    V['2_max_accuracy_diff'] = acc_max

    T_top = st.get('T_top', st['T_done'][-1])
    z = np.load(res / 'checkpoints' / f'T_{T_top}.npz')
    pz = np.load(res / 'predictions' / f'T_{T_top}.npz')
    cells = [str(x) for x in z['cells']]
    keep = [i for i, cl in enumerate(cells) if int(cl.split('_')[0]) not in st['seeds_dropped']]
    if args.replay != 'all':
        rng = np.random.default_rng(19)
        keep = sorted(rng.choice(keep, min(int(args.replay), len(keep)), replace=False).tolist())
    names = [k for k in z.files if k != 'cells']
    model = C.M11.build_model(c11)
    worst_mean = worst_str = 0.0
    for i in keep:
        cl = cells[i]
        tid = cl.split('_', 1)[1].rsplit('_', 1)[0]
        t = tasks[tid]
        model.load_state_dict({k: torch.from_numpy(z[k][i].copy()) for k in names})
        lab = C.D11.labels_for(t, tb, tl)
        ev = C.M11.evaluate(model, t['task_token'], tb, tl, lab)
        g = pz['bits'][i]
        worst_str = max(worst_str, float(np.abs(ev['per_string_bits'] - g).max()))
        worst_mean = max(worst_mean, abs(ev['excess_bits'] - float(g.astype(np.float64).mean())))
    V['3_checkpoints_replayed'] = len(keep)
    V['3_max_abs_diff_mean_bits'] = worst_mean
    V['3_max_abs_diff_string_bits'] = worst_str

    if args.skip_analysis:
        V['5_skipped_smoke_only'] = False
    else:
        check_analysis(res, V)
    ok = (V['1_sources_match_manifest'] and V['4_test_set_identical_to_exp11'] and n_ok == n_rows
          and V['2_rows_equal_csv'] and acc_max <= 1 / 4096 + 1e-12 and worst_mean <= 1e-4
          and all(V[k] for k in V if k.startswith('5_')))
    V['verdict'] = 'PASS' if ok else 'FAIL'
    (res / 'verification.json').write_text(json.dumps(V, indent=1))
    print(json.dumps(V, indent=1))
    sys.exit(0 if ok else 1)


def check_analysis(res, V):
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for f in ['metrics.csv', 'run_state.json']:
            shutil.copy(res / f, tdp / f)
        for st_ in (1, 2):
            subprocess.run([sys.executable, str(HERE / 'analyze.py'), '--stage', str(st_), '--results', td],
                           check=True, capture_output=True)
        subprocess.run([sys.executable, str(HERE / 'summary.py'), '--results', td], check=True, capture_output=True)
        if (res / 'gpu_usage.json').exists():
            shutil.copy(res / 'gpu_usage.json', tdp / 'gpu_usage.json')
            subprocess.run([sys.executable, str(HERE / 'summary.py'), '--results', td], check=True,
                           capture_output=True)
        V['5_forecasts_hash_identical'] = sha(tdp / 'forecasts_frozen.json') == sha(res / 'forecasts_frozen.json')
        A0, A1 = (json.loads((p / 'analysis.json').read_text()) for p in (res, tdp))
        V['5_gates_identical'] = A0['gates'] == A1['gates']
        V['5_E_identical'] = A0['E'] == A1['E']
        V['5_analysis_identical'] = A0 == A1
        V['5_tstar_identical'] = sha(tdp / 'tstar.csv') == sha(res / 'tstar.csv')
        V['5_summary_identical'] = json.loads((tdp / 'summary.json').read_text()) == \
            json.loads((res / 'summary.json').read_text())


if __name__ == '__main__':
    main()
