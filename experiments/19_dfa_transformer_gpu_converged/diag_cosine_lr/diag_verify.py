"""AT-494 diagnostic verification (PREREG.md Section 5). Must print PASS before any number is reported.

  python diag_verify.py --results results

1. Sources (this folder, Experiment 19's expt_v1 files it imports, Experiment 11's files) match manifest.json.
2. Every metrics.csv row is recomputed from predictions/T_<T>.npz (excess bits and per-length bits exactly, accuracy
   within 1/4096), and every lr column equals the schedule formula.
3. The final checkpoint of every cell of every arm is loaded into Experiment 11's model.py on the CPU and evaluated on
   the regenerated test set; per-string bits must match the GPU predictions (mean within 1e-4 bits).
4. The test set is identical to Experiment 11's, and each seed's initialization hash recomputed on the CPU equals
   the one the run recorded (so both arms start exactly where Experiment 19's cells started).
5. diag_summary.py rerun in a scratch copy gives an identical summary.json.
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

import diag_train as DT

C = DT.C
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    args = ap.parse_args()
    res = Path(args.results)
    man = json.loads((res / 'manifest.json').read_text())
    st = json.loads((res / 'run_state.json').read_text())
    V = {}
    V['1_sources_match_manifest'] = all(DT.sha(DT.SOURCES[f] / f) == h for f, h in man['sources'].items()) and \
        all(DT.sha(C.E11 / f) == h for f, h in man['e11_sources'].items())

    c11 = C.load_c11()
    tasks = {t['task_id']: t for t in C.load_tasks()}
    tb, tl = C.D11.build_test_set(c11)
    ref = np.load(C.E11 / 'results' / 'data' / 'dfa_n04_t1_test.npz')
    V['4_test_set_identical_to_exp11'] = bool(np.array_equal(ref['bits'].astype(np.uint64), tb.astype(np.uint64))
                                              and np.array_equal(ref['length'], tl))
    ih = {}
    for s in st['init_sha256']:
        torch.manual_seed(int(s))
        ih[s] = DT.init_hash(C.M11.build_model(c11).state_dict())
    V['4_init_hashes_match'] = ih == st['init_sha256']

    Ls = np.unique(tl)
    rows = list(csv.DictReader(open(res / 'metrics.csv')))
    by = {(r['arm'], r['seed'], r['target_id'], int(r['T'])): r for r in rows}
    n_ok = n_rows = lr_bad = 0
    acc_max = 0.0
    for T in st['T_done']:
        z = np.load(res / 'predictions' / f'T_{T}.npz')
        for i, cell in enumerate(z['cells']):
            arm, s, tid = str(cell).split('_', 2)
            r = by[(arm, s, tid, T)]
            pr = z['bits'][i].astype(np.float64)
            bl = [float(pr[tl == L].mean()) for L in Ls]
            n_ok += float(r['excess_bits']) == float(pr.mean()) and json.loads(r['excess_bits_by_length']) == bl \
                and int(r['target_examples_seen']) == T and int(r['distractor_examples_seen']) == 0
            acc_max = max(acc_max, abs(float(r['accuracy']) - float((z['bits'][i] < 1.0).mean())))
            want = DT.lr_factor(arm, T // c11['batch_size'] - 1, c11['warmup_steps'], st['n_steps']) * c11['lr']
            lr_bad += abs(float(r['lr']) - want) > 1e-15
            n_rows += 1
    V['2_metrics_rows_recomputed'] = f'{n_ok}/{n_rows}'
    V['2_rows_equal_csv'] = n_rows == len(rows)
    V['2_max_accuracy_diff'] = acc_max
    V['2_lr_rows_off_schedule'] = lr_bad

    T_top = st['T_top']
    z = np.load(res / 'checkpoints' / f'T_{T_top}.npz')
    pz = np.load(res / 'predictions' / f'T_{T_top}.npz')
    cells = [str(x) for x in z['cells']]
    assert cells == [str(x) for x in pz['cells']]
    names = [k for k in z.files if k != 'cells']
    model = C.M11.build_model(c11)
    worst_mean = worst_str = 0.0
    for i, cl in enumerate(cells):
        t = tasks[cl.split('_', 2)[2]]
        model.load_state_dict({k: torch.from_numpy(z[k][i].copy()) for k in names})
        ev = C.M11.evaluate(model, t['task_token'], tb, tl, C.D11.labels_for(t, tb, tl))
        g = pz['bits'][i]
        worst_str = max(worst_str, float(np.abs(ev['per_string_bits'] - g).max()))
        worst_mean = max(worst_mean, abs(ev['excess_bits'] - float(g.astype(np.float64).mean())))
    V['3_checkpoints_replayed'] = len(cells)
    V['3_max_abs_diff_mean_bits'] = worst_mean
    V['3_max_abs_diff_string_bits'] = worst_str

    with tempfile.TemporaryDirectory() as td:
        for f in ['metrics.csv', 'run_state.json']:
            shutil.copy(res / f, Path(td) / f)
        subprocess.run([sys.executable, str(DT.HERE / 'diag_summary.py'), '--results', td], check=True,
                       capture_output=True)
        V['5_summary_identical'] = (res / 'summary.json').exists() and \
            json.loads((Path(td) / 'summary.json').read_text()) == json.loads((res / 'summary.json').read_text())

    ok = (V['1_sources_match_manifest'] and V['4_test_set_identical_to_exp11'] and V['4_init_hashes_match']
          and n_ok == n_rows and V['2_rows_equal_csv'] and acc_max <= 1 / 4096 + 1e-12 and lr_bad == 0
          and worst_mean <= 1e-4 and V['5_summary_identical'])
    V['verdict'] = 'PASS' if ok else 'FAIL'
    (res / 'verification.json').write_text(json.dumps(V, indent=1))
    print(json.dumps(V, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
