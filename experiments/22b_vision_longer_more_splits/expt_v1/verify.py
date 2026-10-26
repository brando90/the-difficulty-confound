"""Experiment 22b checks (PROTOCOL Section 5).

  python verify.py --stage remote   # on the cluster, in the post job (GPU): data hashes, every run present, two replays,
                                    # structure + analysis recomputed, results/MANIFEST.sha256 written
  python verify.py --stage local    # here, after fetching results/: manifest, analysis recomputed (1e-9),
                                    # structure recomputed (tolerance), RESULTS.md numbers match analysis.json
Prints one line per check and VERIFY PASS / VERIFY FAIL; exit code 0 only on PASS."""
import argparse, glob, hashlib, json, os, shutil, sys, tempfile
import numpy as np

import data as D
import run as R
import analyze as AN

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAYS = [('vis_real', 0)]  # (phase, index in its run list): the first vision cell
REPLAY_TOL = 0.03   # accuracy; GPU kernels in reduced precision are not bit-reproducible across devices
STRUCT_TOL = 1e-4
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f': {detail}' if detail else ''), flush=True)
    if not ok:
        fails.append(name)


def manifest(results):
    out = {}
    for p in sorted(glob.glob(os.path.join(results, '**', '*'), recursive=True)):
        if os.path.isfile(p) and not p.endswith('MANIFEST.sha256'):
            out[os.path.relpath(p, results)] = D.sha_file(p)
    return out


def close(a, b, tol, path=''):
    if isinstance(a, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol, f'{path}/{k}') for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol, path) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol
    return a == b


def remote(results, pilot=False):
    want = json.load(open(os.path.join(HERE, 'data_sha256.json')))['prepared']
    got = D.prepared_hashes()
    check('data hashes', got == want, ', '.join(k for k in want if got.get(k) != want[k]))
    for ph in R.ORDER:
        lst = os.path.join(HERE, '..', f'runs_{ph}.txt')
        specs = [json.loads(l) for l in open(lst)] if os.path.exists(lst) else R.specs(ph)
        if pilot:
            specs = [dict(c, seed=R.CFG['pilot']['seed']) for c in R.PILOT_CELLS if c['phase'] == ph]
        missing = [R.run_id(s) for s in specs if not os.path.exists(R.out_path(s, results))]
        check(f'runs present {ph}', not missing, f'{len(specs) - len(missing)}/{len(specs)}' + (f' missing {missing[:5]}' if missing else ''))
    errs = glob.glob(os.path.join(results, '*', '*.error'))
    check('no .error files', not errs, ' '.join(os.path.basename(e) for e in errs[:5]))
    tmp = tempfile.mkdtemp(prefix='exp22b_replay_')
    for ph, i in REPLAYS:
        s = R.specs(ph)[i] if not pilot else dict([c for c in R.PILOT_CELLS if c['phase'] == ph][i], seed=R.CFG['pilot']['seed'])
        orig_p = R.out_path(s, results)
        if not os.path.exists(orig_p):
            check(f'replay {R.run_id(s)}', False, 'original missing')
            continue
        R.run_one(s, pilot=pilot, root=tmp)
        a, b = json.load(open(orig_p)), json.load(open(R.out_path(s, tmp)))
        d1 = abs(a['indist_test_acc'] - b['indist_test_acc'])
        d2 = abs(a['heldout_acc'] - b['heldout_acc'])
        check(f'replay {R.run_id(s)}', d1 <= REPLAY_TOL and d2 <= REPLAY_TOL,
              f"in-dist {a['indist_test_acc']:.4f} vs {b['indist_test_acc']:.4f}, held-out {a['heldout_acc']:.4f} vs "
              f"{b['heldout_acc']:.4f}, best epoch {a['best_epoch']} vs {b['best_epoch']}, same preds {a['pred_sha256'] == b['pred_sha256']}")
        shutil.copy(R.out_path(s, tmp), os.path.join(results, f'replay_{R.run_id(s)}.json'))
    import structure as ST
    st = ST.compute(results)
    with open(os.path.join(results, 'structure.json'), 'w') as f:
        json.dump(st, f, indent=1, sort_keys=True)
    check('structure computed', len(st['units']) > 0, f"{len(st['units'])} units")
    an = AN.analyze(results)
    with open(os.path.join(results, 'analysis.json'), 'w') as f:
        json.dump(an, f, indent=1, sort_keys=True)
    check('analysis computed', an['n_runs'] > 0, str(an['runs_by_phase']))
    m = manifest(results)
    with open(os.path.join(results, 'MANIFEST.sha256'), 'w') as f:
        for k, v in m.items():
            f.write(f'{v}  {k}\n')
    check('manifest written', True, f'{len(m)} files')


def local(results, results_md):
    mp = os.path.join(results, 'MANIFEST.sha256')
    if os.path.exists(mp):
        want = dict(reversed(l.rstrip('\n').split('  ', 1)) for l in open(mp) if l.strip())
        got = manifest(results)
        bad = [k for k in want if got.get(k) != want[k]]
        check('manifest', not bad, f'{len(want)} files' + (f', mismatched {bad[:5]}' if bad else ''))
    else:
        check('manifest', False, 'no MANIFEST.sha256')
    an = AN.analyze(results)
    saved = json.load(open(os.path.join(results, 'analysis.json')))
    an = json.loads(json.dumps(an))  # same types as the saved file (tuples -> lists)
    check('analysis recomputed (within 1e-9; numpy/scipy versions differ from the cluster)', close(an, saved, 1e-9))
    try:
        import structure as ST
        st = ST.compute(results)
        old = json.load(open(os.path.join(results, 'structure.json')))
        check('structure recomputed', close(st['units'], old['units'], STRUCT_TOL),
              f"sklearn here {st['sklearn']}, in structure.json {old['sklearn']}")
    except FileNotFoundError as e:
        check('structure recomputed', False, f'data missing here: {e}')
    if results_md and os.path.exists(results_md):
        txt = open(results_md).read()
        miss = []
        for arm, x in saved['by_arm'].items():
            for k, v in x['verdicts'].items():
                if f'{k} {v}' not in txt:
                    miss.append(f'{arm} {k} {v}')
            for key in ('P1_shuffled_p0_transfer', 'real_p0_transfer'):
                if x[key]['mean'] is not None and f"{x[key]['mean']:.3f}" not in txt:
                    miss.append(f'{arm} {key}')
        check('RESULTS.md matches analysis.json', not miss, ', '.join(miss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['remote', 'local'], required=True)
    ap.add_argument('--results')
    ap.add_argument('--pilot', action='store_true', help='check the code-path pilot in results_pilot/ (replays at pilot size)')
    ap.add_argument('--results-md', default=os.path.join(HERE, '..', 'RESULTS.md'))
    a = ap.parse_args()
    results = a.results or os.path.join(HERE, 'results_pilot' if a.pilot else 'results')
    remote(results, a.pilot) if a.stage == 'remote' else local(results, a.results_md)
    print('VERIFY PASS' if not fails else f'VERIFY FAIL ({len(fails)}): {fails}')
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
