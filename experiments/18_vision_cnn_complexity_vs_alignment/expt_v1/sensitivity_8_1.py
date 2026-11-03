"""PROTOCOL Section 8.1 sensitivity: replace the 3 remote probe mismatches with the Intel recompute and rerun the H analysis."""
import json, os, sys, tempfile
E = os.path.dirname(os.path.abspath(__file__))
T = tempfile.mkdtemp()
for arm in ('main', 'H', 'mis'):
    os.makedirs(f'{T}/{arm}')
    for f in os.listdir(f'{E}/results/{arm}'):
        if f.endswith('.json'):
            os.symlink(f'{E}/results/{arm}/{f}', f'{T}/{arm}/{f}')
sub = {'H__kmnist__pool_only__easy__none__s18202': 0.75830078125,
       'H__kmnist__pool_only__hard_mod__none__s18303': 0.985595703125,
       'H__cifar10__pool_only__hard_mod__none__s18202': 0.980712890625}
for rid, v in sub.items():
    p = f'{T}/H/{rid}.json'
    d = json.load(open(p)); os.remove(p)
    assert abs(d['rows'][-1]['probe_err__a7b'] - v) < 0.01
    r = d['rows'][-1]
    r['probe_eps__a7b'] = r['probe_eps__a7b'] * v / r['probe_err__a7b']  # the analysis reads eps = err / chance
    r['probe_err__a7b'] = v
    json.dump(d, open(p, 'w'))
sys.path.insert(0, E)
import analyze as AN
A = json.load(open(f'{E}/results/analysis.json'))
B = {'H': {ds: json.loads(json.dumps(AN.analyse_H(T, ds), default=float)) for ds in ('kmnist', 'cifar10')}}
# control: the same function on the unmodified results reproduces analysis.json
C = {ds: json.loads(json.dumps(AN.analyse_H(f'{E}/results', ds), default=float)) for ds in ('kmnist', 'cifar10')}
out = {}
for ds in ('kmnist', 'cifar10'):
    for mode in ('joint', 'probe'):
        a, b = A['H'][ds][mode], B['H'][ds][mode]
        out[f'{ds}/{mode}'] = dict(GH1=[a['GH1'], b['GH1']], GH2=[a['GH2'], b['GH2']],
                                   rho_HE_S=[a['S']['rho_HE']['ratio'], b['S']['rho_HE']['ratio']],
                                   rho_EH_S=[a['S']['rho_EH']['ratio'], b['S']['rho_EH']['ratio']],
                                   rho_EH_last=[a['last']['rho_EH']['ratio'], b['last']['rho_EH']['ratio']],
                                   identical=a == b)
    out[f'{ds}/control_reproduces_analysis_json'] = C[ds] == A['H'][ds]
out['note'] = 'Only H runs carry probe errors; the main and mismatch analyses do not read them.'
json.dump(out, open(f'{E}/results/sensitivity_8_1.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
