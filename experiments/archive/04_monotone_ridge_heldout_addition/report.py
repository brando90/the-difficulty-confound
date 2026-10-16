"""Post-processing only: describe every frozen outcome, with no selection/refits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expt_v1/results'
OLD=ROOT.parent/'archive/03_answer_matched_alignment/expt_v1/results'
if not OLD.exists():OLD=ROOT.parent/'03_answer_matched_alignment/expt_v1/results'

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def key(r):return tuple(float(r[k]) for k in ['seed','width','mixture','steps','carry_chain'])

def matrix(rows,names):
    mapping={'log_parameters':'parameter_count','log_updates':'steps','log_exposure':'stratum_fraction','log_starting_bits':'starting_bits'}
    return np.array([[1. if n=='intercept' else float(r['carry_chain'])==2 if n=='two_carry' else np.log(float(r[mapping[n]])) for n in names] for r in rows],float)

def diagnosis():
    f=json.loads((OLD/'frozen_forecasts.json').read_text());cal={key(r):r for r in read(OLD/'calibration_metrics.csv')}
    rows=[r for r in read(OLD/'scored_forecasts.csv') if r['law']=='starting_difficulty' and r['axis']=='joint']
    total=sum(float(r['error'])**2 for r in rows);diag=[]
    for fit in f['fits']:
        if fit['law']!='starting_difficulty':continue
        fr=[cal[tuple(k)] for k in fit['training_keys']];rr=[r for r in rows if float(r['seed'])==fit['seed']]
        x=matrix(fr,fit['columns']);beta=np.array(fit['coefficients']);coef=np.linalg.lstsq(x,np.log([float(r['mean_bits']) for r in fr]),rcond=None)[0]
        assert np.allclose(coef,beta,atol=1e-10,rtol=0)
        computed=np.exp(matrix([cal[key(r)] for r in rr],fit['columns'])@coef)
        assert np.allclose(computed,[float(r['predicted_bits']) for r in rr],atol=1e-9,rtol=1e-12)
        z=x.copy();z[:,1:]=(x[:,1:]-x[:,1:].mean(0))/x[:,1:].std(0)
        worst=max(rr,key=lambda r:abs(float(r['error'])));errors=np.array([float(r['error']) for r in rr])
        diag.append(dict(seed=fit['seed'],difficulty_exponent=float(beta[-1]),fit_d_min=min(float(r['starting_bits']) for r in fr),fit_d_max=max(float(r['starting_bits']) for r in fr),target_d_min=min(float(cal[key(r)]['starting_bits']) for r in rr),target_d_max=max(float(cal[key(r)]['starting_bits']) for r in rr),joint_rmse=float(np.sqrt(np.mean(errors**2))),squared_error_share=float(np.sum(errors**2)/total),rank=int(np.linalg.matrix_rank(x)),raw_condition=float(np.linalg.cond(x)),standardized_condition=float(np.linalg.cond(z)),worst_width=float(worst['width']),worst_mixture=float(worst['mixture']),worst_steps=float(worst['steps']),worst_carry=float(worst['carry_chain']),worst_prediction=float(worst['predicted_bits']),worst_observation=float(worst['observed_bits'])))
    write(OUT/'prior_failure_diagnosis.csv',diag)
    return diag

def fmt(x):return f'{float(x):.4f}'
def ci(r,base='rmse'):return f"{fmt(r[base])} [{fmt(r[base+'_lo'])}, {fmt(r[base+'_hi'])}]"

def main():
    c=json.loads((ROOT/'expt_v1/config.json').read_text());rt=json.loads((OUT/'runtime.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['status']=='PASS'
    p=json.loads((OUT/'prediction_summary.json').read_text());a=json.loads((OUT/'intervention_summary.json').read_text());diag=diagnosis()
    scored=read(OUT/'scored_forecasts.csv');fits=json.loads((OUT/'frozen_forecasts.json').read_text())['fits']
    seedstats=[]
    for law in sorted({r['law'] for r in scored}):
        for seed in c['seeds']:
            rr=[r for r in scored if r['axis']=='joint' and r['law']==law and float(r['seed'])==seed];err=np.array([float(r['error']) for r in rr]);worst=max(rr,key=lambda r:abs(float(r['error'])))
            seedstats.append(dict(law=law,seed=seed,n=len(rr),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),bias=float(err.mean()),worst_prediction=float(worst['predicted_bits']),worst_observation=float(worst['observed_bits']),worst_width=float(worst['width']),worst_mixture=float(worst['mixture']),worst_steps=float(worst['steps']),worst_carry=float(worst['carry_chain'])))
    write(OUT/'seed_prediction_metrics.csv',seedstats)
    coefficient_rows=[]
    for fit in fits:
        coefficient_rows.append({'seed':fit['seed'],'law':fit['law'],**dict(zip(fit['columns'],fit['coefficients']))})
    for r in coefficient_rows:r.setdefault('log_starting_bits','')
    write(OUT/'coefficient_table.csv',coefficient_rows)
    unions=[]
    for seed in c['seeds']:
        trains=np.load(OUT/f'train_{seed}.npz');warm=set(trains['q0.50'].tolist())
        for q in c['train_mixtures']+c['heldout_mixtures']:
            ids=set(trains[f'q{q:.2f}'].tolist());unions.append(dict(seed=seed,mixture=q,intervention_unique=len(ids),warmup_unique=len(warm),combined_unique=len(ids|warm),intersection=len(ids&warm)))
        endpoints=[r['combined_unique'] for r in unions if r['seed']==seed and r['mixture'] in [.1,.9]];assert endpoints==[6552,6552]
    write(OUT/'warmup_union_counts.csv',unions)
    marginals=read(OUT/'input_digit_marginals.csv');tv=[]
    for seed in c['seeds']:
        for col in range(6):
            def hist(q):return np.array([int(r['count'])/int(r['n']) for r in marginals if int(r['seed'])==seed and int(r['column'])==col and float(r['mixture'])==q])
            tv.append(dict(seed=seed,column=col,total_variation=float(abs(hist(.1)-hist(.9)).sum()/2)))
    write(OUT/'input_marginal_total_variation.csv',tv)
    main=p['primary_secondary_comparison'];repair=p['repair_comparison'];joint=[r for r in p['metrics'] if r['axis']=='joint']
    lines=['# Fresh held-out addition: extrapolation repair and feature value','',f'**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/04_monotone_ridge_heldout_addition/RESULTS_stable.md>','',f"**TLDR:** Repair criterion: **{p['repair_status']}**. Starting-difficulty feature criterion: **{p['status']}**. All 200 trajectories completed; all 3,840 forecasts and failed criteria are retained.",'','## Setup and decisions','',
    'Canonical experiment: `experiments/04_monotone_ridge_heldout_addition/`. Exact three-digit addition and the byte-identical arithmetic reference remain fixed. New seeds, mixtures .2/.8, widths 96/384, and evaluation items exclude every experiment-03 evaluation/probe item. Training still matches 117 answer frequencies exactly, 4,680 pairs per intervention. There are 1,872 final-test questions (936 per carry stratum), 936 calibration questions and 936 separate probe questions. Input-digit marginals remain unmatched. This is supervised learning, not reinforcement learning.', '',
    'All five laws use model size, updates, carry group and exposure. The three difficulty laws add independently probed starting loss. The old laws use unregularized least squares on log loss; ridge adds a fixed 0.01 standardized-slope penalty. Stable laws also constrain size/updates/exposure slopes to be nonpositive and the difficulty exponent to [0,1]. Those settings were frozen before training. Fit only 54 calibration rows per seed; freeze predictions before opening final-test outcomes.', '',
    'Primary feature-value gate: stable difficulty versus stable base. Separate repair gate: stable difficulty versus legacy difficulty. Each requires at least 10% relative joint root mean squared error reduction, a positive 95% lower interval, improvement at both held-out widths, and no invalid predictions. The 10% threshold and coefficient bounds are project choices. A repair success does not substitute for feature-value success.', '',
    'All intervals below are 95% percentile intervals from 20,000 paired bootstrap draws over eight complete seed blocks, conditional on this fixed test set; p-val=n/a. The 192 joint targets per law are correlated observations, not 192 independent replicates. Scores are bits per answer under the factorized model. Mean bias is forecast minus observed; positive means overprediction. No trained model or seed was discarded.', '',
    f"Endpoint warmup-plus-intervention unions are exactly 6,552 unique pairs per seed in both conditions ([counts](expt_v1/results/warmup_union_counts.csv)); intermediate unions differ. Across six operand-digit columns and eight seeds, endpoint input-marginal total variation ranges from {min(r['total_variation'] for r in tv):.3f} to {max(r['total_variation'] for r in tv):.3f} ([all values](expt_v1/results/input_marginal_total_variation.csv)); zero would mean matching.", '', '## Joint held-out prediction','', '| Law | Root mean squared error [95% interval] | Mean absolute error | Mean bias | Forecast mean | Observed mean | Invalid / targets |','|---|---:|---:|---:|---:|---:|---:|']
    for r in joint:lines.append(f"| {r['law']} | {ci(r)} | {fmt(r['mae'])} | {float(r['bias']):+.4f} | {fmt(r['prediction_mean'])} | {fmt(r['outcome_mean'])} | {r['invalid_predictions']}/{r['n_targets']} |")
    lines+=['','Relative error reduction is 100 × (1 − extended error / baseline error); positive is better.','', '| Baseline → extended | Reduction %, 95% interval |','|---|---:|']
    for r in p['comparisons']:
        if r['axis']=='joint':lines.append(f"| {r['baseline']} → {r['extended']} | {100*r['relative_rmse_reduction']:.2f} [{100*r['lo']:.2f}, {100*r['hi']:.2f}] |")
    lines+=['','## Every held-out split','', '| Split | Law | Targets | Root mean squared error [95% interval] |','|---|---|---:|---:|']
    for r in p['metrics']:lines.append(f"| {r['axis']} | {r['law']} | {r['n_targets']} | {ci(r)} |")
    lines+=['','## Seed variability and worst forecasts','', 'Eight training seeds, one trajectory per seed/width/mixture; no repeated scoring noise was introduced. The following sample standard deviations describe variation of per-seed joint error, not uncertainty from independent test sets.','', '| Law | Mean seed error | Seed standard deviation | Minimum | Maximum | Worst prediction / observed |','|---|---:|---:|---:|---:|---:|']
    for law in sorted({r['law'] for r in seedstats}):
        rr=[r for r in seedstats if r['law']==law];x=np.array([r['rmse'] for r in rr]);worst=max(rr,key=lambda r:abs(r['worst_prediction']-r['worst_observation']))
        lines.append(f"| {law} | {x.mean():.4f} | {x.std(ddof=1):.4f} | {x.min():.4f} | {x.max():.4f} | {worst['worst_prediction']:.4f} / {worst['worst_observation']:.4f} |")
    lines+=['','[Every seed and worst-target key](expt_v1/results/seed_prediction_metrics.csv) · [Every coefficient](expt_v1/results/coefficient_table.csv) · [All split comparisons](expt_v1/results/prediction_comparisons.csv).','','## Secondary ranking-reversal replication','',f"Frozen replication criterion: **{a['status']}**. The endpoint advantages must each exceed 10 percentage points, have a positive interval lower bound, and remain positive at every width.",'','| Contrast | Percentage points [95% interval] |','|---|---:|']
    for name,r in a['statistics'].items():lines.append(f"| {name} | {100*r['estimate']:.2f} [{100*r['lo']:.2f}, {100*r['hi']:.2f}] |")
    lines+=['','`e_low` is no-carry minus two-carry accuracy with 90% no-carry exposure; `e_high` reverses that difference with 90% two-carry exposure. Alignment is their average; structure is half their difference. These are intervention-specific accuracy contrasts, not a universal variance decomposition.','', '| Width | No-carry advantage at 90% exposure | Two-carry advantage at 90% exposure |','|---|---:|---:|']
    for w,r in a['by_width'].items():lines.append(f"| {w} | {100*r['e_low']:.2f} | {100*r['e_high']:.2f} |")
    test=read(OUT/'test_metrics.csv');draws=np.random.default_rng(c['analysis_seed']).integers(0,8,(c['bootstrap_draws'],8));absolute=[]
    lines+=['','Final exact-answer accuracies, pooled equally across widths within seed:','','| Two-carry training fraction | Carry stratum | Accuracy %, 95% interval |','|---|---:|---:|']
    for q in [.1,.5,.9]:
        for cc in [0,2]:
            values=np.array([np.mean([int(r['correct'])/int(r['n']) for r in test if int(r['seed'])==seed and float(r['mixture'])==q and int(r['steps'])==3000 and int(r['carry_chain'])==cc]) for seed in c['seeds']])
            lo,hi=np.quantile(values[draws].mean(1),[.025,.975]);r=dict(mixture=q,carry_chain=cc,accuracy=float(values.mean()),lo=float(lo),hi=float(hi),seed_sd=float(values.std(ddof=1)),seeds=8,widths=5,items_per_stratum=936);absolute.append(r)
            lines.append(f"| {q} | {cc} | {100*r['accuracy']:.2f} [{100*lo:.2f}, {100*hi:.2f}] |")
    write(OUT/'absolute_accuracy.csv',absolute)
    lines+=['','## Why experiment 03 blew up','', 'The fitted form is predicted loss = exp(intercept) × parameters^b × updates^c × exp(d × carry indicator) × exposure^e × starting_loss^f. A negative f makes a better starting loss raise the forecast. This is a post hoc diagnosis of already-published development data. It does not alter experiment 03 or choose settings from the new test. Independent reconstruction reproduces its coefficients and every joint difficulty-law forecast. The raw design is full rank, but correlated size/difficulty features permit inverse powers that behave badly outside the fitting support. Positive forecasts alone do not prevent this.', '', '| Seed | Difficulty exponent | Fit starting-loss range | Joint target range | Joint error | Share of squared error % | Standardized condition number |','|---|---:|---:|---:|---:|---:|---:|']
    for r in diag:lines.append(f"| {r['seed']} | {r['difficulty_exponent']:.4f} | {r['fit_d_min']:.3f}–{r['fit_d_max']:.3f} | {r['target_d_min']:.3f}–{r['target_d_max']:.3f} | {r['joint_rmse']:.4f} | {100*r['squared_error_share']:.5f} | {r['standardized_condition']:.2f} |")
    r=max(diag,key=lambda r:r['squared_error_share'])
    lines+=['',f"Seed {r['seed']} contributes {100*r['squared_error_share']:.5f}% of squared error. Its worst forecast is {r['worst_prediction']:.6f} versus {r['worst_observation']:.6f} observed, width {r['worst_width']:g}, mixture {r['worst_mixture']}, step {r['worst_steps']:g}, carry {r['worst_carry']:g}. The repair blocks this specific inverse-difficulty mechanism by construction; held-out accuracy must still be measured.",'','## Verification, provenance and limitations','',
    f"Training completed in {rt['elapsed_seconds']/60:.2f} minutes on {rt['host']} using two CPU threads; no graphics processor or provider calls for model training. Frozen source: `{rt['git_revision']}`. The original 90-minute cap was preserved. A slow shared-filesystem checkout was replaced by a node-local checkout before training. The Codex executor failed authentication and the personal Claude executor hit its weekly limit before scientific work; the master adopted the existing fixed numerical procedure. Their logs are retained, with no claim that all accounts were exhausted.", '',
    f"Independent verification: {v['oracle_pairs']:,} oracle pairs, {v['aggregate_rows']:,} aggregates reconstructed from {v['item_rows']:,} item observations; {v['final_models_replayed']} final and {v['warm_models_replayed']} warm models replayed, maximum loss discrepancy {v['max_replay_bits_difference']}; {v['forecasts_reconstructed']:,} forecasts reconstructed, maximum discrepancy {v['max_forecast_difference']:.3g}. Independent active-set enumeration agrees with every constrained fit, bounds are checked, bootstrap results match, source/input hashes match. [Receipt](expt_v1/results/verification.json).", '',
    'The study tests these fixed predictive forms on one addition setup. Mixture holdout is interpolation; width 384 tests extrapolation beyond fitting widths. Starting loss is probe-derived and itself depends on capacity and warmup. New item holdout is disjoint from experiment-03 evaluation; new training may contain old evaluation items because these are fresh networks. This does not validate Varun’s original mathematics law, eliminate computational complexity, or establish that data explains most difficulty across tasks. Answer matching does not match input marginals or prove invariant internal strategies. Coefficient shape restrictions are inductive assumptions that can underfit.', '',
    '[Protocol](PROTOCOL.md) · [Live/completion record](results.md) · [Prior full report](../archive/03_answer_matched_alignment/RESULTS_matched_alignment.md) · [Google companion](https://docs.google.com/document/d/19yNEZTimoCPr9H4D_P-53nN83dBNFz_LY-aFdRa6_Bk/edit).', '',
    '![Joint forecast error and ranking reversal](stable_results.png)', '',
    'Figure: left, all five joint prediction errors with paired-seed 95% intervals on a logarithmic axis; right, final accuracy differences under the two endpoint exposure interventions. Points are means across eight seeds at each width. Full tables and failed criteria appear above.']
    (ROOT/'RESULTS_stable.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    names=[r['law'].replace('_','\n') for r in joint];val=np.array([r['rmse'] for r in joint]);lo=np.array([r['rmse_lo'] for r in joint]);hi=np.array([r['rmse_hi'] for r in joint]);ax[0].errorbar(range(5),val,yerr=[val-lo,hi-val],fmt='o',capsize=4,color='#235a83');ax[0].set_xticks(range(5),names,fontsize=8);ax[0].set_yscale('log');ax[0].set_ylabel('Root mean squared error (bits / answer)');ax[0].set_title('Fresh joint holdout: 192 targets / law')
    widths=sorted(map(int,a['by_width']));ax[1].plot(widths,[100*a['by_width'][str(w)]['e_low'] for w in widths],'o-',label='No-carry advantage: 90% no-carry training');ax[1].plot(widths,[100*a['by_width'][str(w)]['e_high'] for w in widths],'s-',label='Two-carry advantage: 90% two-carry training');ax[1].axhline(0,color='black',lw=.7);ax[1].axhline(10,color='gray',ls=':',lw=.8,label='Pooled 10-point gate (not per-width)');ax[1].set_xlabel('Hidden width');ax[1].set_ylabel('Accuracy advantage (percentage points)');ax[1].set_title('Same addition task; exposure changes');ax[1].legend(fontsize=7)
    fig.savefig(ROOT/'stable_results.png',dpi=170);fig.savefig(ROOT/'stable_results.pdf');plt.close(fig)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()}
    (ROOT/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(repair=p['repair_status'],feature=p['status'],rank=a['status'],artifact_files=len(manifest))))

if __name__=='__main__':main()
