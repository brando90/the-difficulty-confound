"""Render complete, verified results; no training or model selection."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from addition import write_csv, save_json
from analyze import read, boot_draws
ROOT=Path(__file__).resolve().parent
LABELS={'resources':'Resources','structure':'Resources + carry','exposure':'Resources + exposure','structure_exposure':'Resources + carry + exposure','starting_difficulty':'Resources + carry + exposure + starting difficulty'}


def main():
    out=ROOT/'results';c=json.loads((ROOT/'config.json').read_text());v=json.loads((out/'verification.json').read_text());assert v['status']=='PASS'
    rt=json.loads((out/'runtime.json').read_text());it=json.loads((out/'intervention_summary.json').read_text());pr=json.loads((out/'prediction_summary.json').read_text());test=read(out/'test_metrics.csv');draws=boot_draws(c)
    widths=sorted(c['train_widths']+c['heldout_widths']);qs=sorted(c['train_mixtures']+c['heldout_mixtures']);table=[]
    for step in c['checkpoints']:
        for w in widths:
            for q in qs:
                for carry in (0,2):
                    rr=sorted([r for r in test if (r['steps'],r['width'],r['mixture'],r['carry_chain'])==(step,w,q,carry)],key=lambda r:r['seed']);assert len(rr)==8
                    acc=np.array([r['correct']/r['n'] for r in rr]);bits=np.array([r['mean_bits'] for r in rr]);ci=np.quantile(acc[draws].mean(1),[.025,.975])
                    table.append(dict(width=w,mixture=q,steps=step,carry_chain=carry,n_seeds=8,n_items_per_seed=936,accuracy=float(acc.mean()),accuracy_lo=float(ci[0]),accuracy_hi=float(ci[1]),accuracy_seed_sd=float(acc.std(ddof=1)),mean_bits=float(bits.mean()),bits_seed_sd=float(bits.std(ddof=1))))
    write_csv(out/'accuracy_summary.csv',table)
    fig,axes=plt.subplots(1,5,figsize=(16,3.9),sharex=True,sharey=True)
    for ax,w in zip(axes,widths):
        for carry,color,label in [(0,'#2166ac','No carry'),(2,'#d6604d','Two carries')]:
            rows=[r for r in table if r['steps']==3000 and r['width']==w and r['carry_chain']==carry]
            ax.plot([r['mixture'] for r in rows],[100*r['accuracy'] for r in rows],marker='o',color=color,label=label)
            ax.fill_between([r['mixture'] for r in rows],[100*r['accuracy_lo'] for r in rows],[100*r['accuracy_hi'] for r in rows],color=color,alpha=.14)
        ax.set_title(f'Width {w}');ax.set_xticks(qs);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Exact-answer accuracy (%)');axes[2].set_xlabel('Two-carry fraction in intervention training pool');axes[-1].legend(loc='lower right',frameon=False)
    fig.suptitle('Same addition task and answer histogram; changing training exposure',fontsize=14)
    fig.tight_layout();fig.savefig(out/'alignment_accuracy.png',dpi=180);fig.savefig(out/'alignment_accuracy.pdf');plt.close(fig)
    endpoints=[]
    for q in (.1,.5,.9):
        for carry in (0,2):
            values=np.array([np.mean([r['correct']/r['n'] for r in test if (r['steps'],r['seed'],r['mixture'],r['carry_chain'])==(3000,seed,q,carry)]) for seed in c['seeds']])
            lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
            endpoints.append(dict(mixture=q,carry_chain=carry,accuracy=float(values.mean()),lo=float(lo),hi=float(hi),seed_sd=float(values.std(ddof=1)),n_seeds=8,n_widths=5,n_items_per_stratum=936))
    write_csv(out/'endpoint_accuracy.csv',endpoints)
    sw=read(out/'intervention_seed_width.csv');variability={}
    for key in ['e_low','e_high','alignment','structure','balanced_gap']:
        values=[np.mean([r[key] for r in sw if r['seed']==seed]) for seed in c['seeds']];variability[key]=float(np.std(values,ddof=1))
    save_json(out/'intervention_seed_variability.json',variability)
    unionrows=[]
    for seed in c['seeds']:
        z=np.load(out/f'train_{seed}.npz');warm=z['q0.50']
        for q in qs:
            n=len(np.union1d(warm,z[f'q{q:.2f}']));expected={.1:6552,.3:5616,.5:4680,.7:5616,.9:6552}[q];assert n==expected
            unionrows.append(dict(seed=seed,mixture=q,intervention_pairs=4680,warmup_pairs=4680,union_pairs=n))
    write_csv(out/'warmup_union_counts.csv',unionrows)
    marginal=read(out/'input_digit_marginals.csv');tvrows=[]
    for seed in c['seeds']:
        for col in range(6):
            def freq(q):return np.array([r['count']/r['n'] for r in sorted([r for r in marginal if (r['seed'],r['column'],r['mixture'])==(seed,col,q)],key=lambda r:r['digit'])])
            tvrows.append(dict(seed=seed,column=col,total_variation=float(.5*np.sum(abs(freq(.1)-freq(.9))))))
    write_csv(out/'input_marginal_distances.csv',tvrows)
    tvmeans=[float(np.mean([r['total_variation'] for r in tvrows if r['column']==col])) for col in range(6)]
    def pp(z):return f"{100*z['estimate']:.2f} [{100*z['lo']:.2f}, {100*z['hi']:.2f}]"
    stats=it['statistics'];gain=pr['primary_secondary_comparison'];gaintext=f"{100*gain['relative_rmse_reduction']:.2f}% [{100*gain['lo']:.2f}, {100*gain['hi']:.2f}]"
    url='https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/03_answer_matched_alignment'
    lines=['# Answer-matched alignment: complete results','',f'**Doc link:** <{url}/RESULTS_matched_alignment.md>','',f"**TLDR:** The prespecified rank-reversal criterion is **{it['status']}**. Adding starting difficulty to the structure-and-exposure prediction law changes joint held-out root mean squared error by a relative reduction of **{gaintext}**, so that separate criterion is **{pr['status']}**.",'',
    '## What was tested','',
    'All 200 trajectories and 40 shared warm starts completed: eight paired seeds, five widths, five training mixtures, three fixed checkpoints. Every condition adds the same three-digit integers with the same reference and notation; only the intervention training distribution changes within each width. Every intervention has 4,680 distinct training pairs and exactly 40 examples for each of 117 answer values. Evaluation has 1,872 fixed questions (936 per carry stratum); calibration and independent starting-difficulty probes each have 936 questions. No pairs or swapped pairs cross splits. Both evaluation strata have identical exact-answer histograms.','',
    'The warmup-plus-intervention pool is 6,552 pairs at both primary endpoints; intermediate mixtures have smaller unions. The 9:1 ratio describes the intervention, after 500 balanced warmup updates. [Accounting and remaining input confounds](SAMPLING_NOTES.md). This restricted population uses no-carry and two-carry examples with totals below 1000; it is not the full addition population from the earlier studies.','',
    '## Primary intervention result','',
    '| Contrast at 3,000 intervention updates | Mean [95% interval], percentage points | Seed standard deviation, points |','|---|---:|---:|']
    for name,label in [('e_low','No-carry advantage with 90% no-carry training'),('e_high','Two-carry advantage with 90% two-carry training'),('alignment','Alignment contrast A'),('structure','Residual carry main effect S'),('balanced_gap','No-carry advantage under balanced training')]:lines.append(f"| {label} | {pp(stats[name])} | {100*variability[name]:.2f} |")
    lines += [f"| A − absolute S | {pp(stats['alignment_minus_abs_structure'])} | — |",'',
    f"Verdict: **{it['status']}**. The frozen criterion requires both endpoint advantage means at least 10 points, both lower interval bounds above zero, and both advantages positive at each width. Ten points is a prespecified project practical margin, not a literature threshold. Intervals resample eight whole paired seed blocks 20,000 times, averaging widths within seed; they condition on one fixed test set. p-val=n/a: decision uses the interval gate, with no separate significance test. The A/S comparison applies only to this specified intervention and accuracy scale.",'',
    '| Width | No-carry advantage at q=.1, points | Two-carry advantage at q=.9, points | Balanced no-carry gap, points |','|---|---:|---:|---:|']
    for w in widths:
        r=it['by_width'][str(w)];lines.append(f"| {w} | {100*r['e_low']:.2f} | {100*r['e_high']:.2f} | {100*r['balanced_gap']:.2f} |")
    lines += ['', '| Two-carry training fraction | Evaluation stratum | Accuracy [95% interval], % | Seed standard deviation, points |','|---|---|---:|---:|']
    for r in endpoints:
        label='No carry' if r['carry_chain']==0 else 'Two carries'
        lines.append(f"| {r['mixture']:.1f} | {label} | {100*r['accuracy']:.2f} [{100*r['lo']:.2f}, {100*r['hi']:.2f}] | {100*r['seed_sd']:.2f} |")
    lines += ['', 'Accuracy averages five widths within each of eight paired seeds; every stratum has the same 936 fixed evaluation items. [Machine-readable endpoint accuracies](expt_v1/results/endpoint_accuracy.csv).','', '![Accuracy by training mixture and network width](expt_v1/results/alignment_accuracy.png)','',
    'Shading: 95% paired-seed intervals at each point. Every checkpoint, width, mixture and stratum is retained in [accuracy_summary.csv](expt_v1/results/accuracy_summary.csv); no favorable checkpoint was selected.','',
    '## Prospective prediction result','',
    'Five fixed log-linear laws fit log calibration loss separately within each seed and exponentiate forecasts, guaranteeing positive predictions. Fits use only widths 32/64/128 and mixtures .1/.5/.9; widths 96/256 and mixtures .3/.7 are held out. Fifty-four calibration cells fit each seed; joint evaluation has 192 targets per law (eight seeds × two widths × two mixtures × three checkpoints × two strata). Starting difficulty is loss on a separate probe after common warmup, before intervention. Coefficients and all 3,840 forecasts were saved before the analysis opened final-test outcomes.','',
    'These power-law models test the portability of a starting-difficulty feature. They change the functional form from the previous additive law and do not constitute a fresh test on Varun’s original mathematics data.','',
    '| Joint held-out law | Root mean squared error [95% interval], bits/answer | Mean absolute error | Bias | Forecast mean | Observed mean |','|---|---:|---:|---:|---:|---:|']
    for r in pr['metrics']:
        if r['axis']=='joint':lines.append(f"| {LABELS[r['law']]} | {r['rmse']:.4f} [{r['rmse_lo']:.4f}, {r['rmse_hi']:.4f}] | {r['mae']:.4f} | {r['bias']:+.4f} | {r['prediction_mean']:.4f} | {r['outcome_mean']:.4f} |")
    lines += ['',f"Starting-difficulty criterion: **{pr['status']}**, relative error reduction **{gaintext}**. Required: at least 10% improvement, a positive lower interval bound, and a positive improvement at both held-out widths. Positive means better; negative means worse. All forecasts were finite and positive. Bias is forecast minus observation; positive means overestimated loss. Intervals resample the eight paired seed blocks; targets within a seed are not independent. p-val=n/a.",'',
    '| Split | Added feature(s) | Relative error reduction [95% interval], % | Targets per law |','|---|---|---:|---:|']
    for r in pr['comparisons']:
        feature='Starting difficulty beyond structure + exposure' if r['extended']=='starting_difficulty' else 'Exposure beyond structure' if r['baseline']=='structure' else 'Structure beyond exposure'
        lines.append(f"| {r['axis']} | {feature} | {100*r['relative_rmse_reduction']:.2f} [{100*r['lo']:.2f}, {100*r['hi']:.2f}] | {r['n_targets']} |")
    lines += ['', 'All splits and laws: [metrics](expt_v1/results/prediction_metrics.csv), [comparisons](expt_v1/results/prediction_comparisons.csv), [scored forecasts](expt_v1/results/scored_forecasts.csv), [fit coefficients and training IDs](expt_v1/results/frozen_forecasts.json). The incremental exposure and structure contrasts are secondary; no multiple-comparison-adjusted claim is made.','',
    '## Complexity and construct limits','',
    'Input marginal audit: total variation distance (half the absolute differences summed over ten digits) between endpoint training distributions, averaged across eight seeds, is '+', '.join(f'{x:.4f}' for x in tvmeans)+' for the six input positions (little-endian digits of the first operand, then the second). Zero would mean identical digit marginals; these values quantify the remaining mismatch. [All seed/position distances](expt_v1/results/input_marginal_distances.csv).','',
    'Carry-chain length describes a structural demand of standard column addition; it is not Kolmogorov complexity or an asymptotic runtime lower bound. For an identical evaluation item and fixed reference machine, changing training data does not change the shortest-program description of its answer conditional on the input and task. This invariance requires no computation of that uncomputable quantity. [Detailed distinction and sources](../01_fixed_width_addition_exposure/COMPLEXITY.md).','',
    'A ranking reversal would show that carry structure alone does not determine this learner’s observed difficulty. It would not show that complexity is irrelevant or that data explains most real-world language-model performance. Exact answer matching leaves input-digit marginals and operand distributions unmatched; the intervention changes a whole distribution, not an isolated carry mechanism. The fixed task and evaluator do not establish identical internal strategies. Starting difficulty can encode capacity, optimization and structure as well as familiarity; predictive usefulness is not proof of a data-only mechanism. Conditional answer bits omit model/data transmission costs and are not a full minimum-description-length score.','',
    '## Verification and reproduction','',
    f"Training: {rt['elapsed_seconds']:.2f} seconds ({rt['elapsed_seconds']/60:.2f} minutes), two CPU threads on {rt['host']}; no graphics processor or model-provider calls. Source revision `{rt['git_revision']}`. Protocol commit `34dc473`; no outcome-driven scientific changes or retries.",'',
    f"Verification **PASS**: {v['oracle_pairs']:,} reference pairs checked, {v['aggregate_rows']:,} aggregate rows recomputed from {v['item_rows']:,} per-item rows, {v['final_models_replayed']} final weights and {v['warm_models_replayed']} warm weights replayed. Maximum final replay loss difference {v['max_replay_bits_difference']}; all {v['forecasts_reconstructed']:,} forecasts independently reconstructed; maximum forecast difference {v['max_forecast_difference']:.3g}. Paired bootstrap comparisons matched independently. [Machine-readable verification](expt_v1/results/verification.json).",'',
    'Use Python with NumPy, PyTorch and Matplotlib. Exact runtime versions and source hashes are in [runtime.json](expt_v1/results/runtime.json). From this experiment folder:', '',
    '```bash','python expt_v1/verify.py','python expt_v1/analyze.py','python expt_v1/report.py','```','',
    'Training reproduction: run `python expt_v1/run.py --check-only`; for a new execution, copy the frozen source/config into a fresh output directory first. The driver refuses to overwrite an existing run. The complete archived run includes weights, training IDs, per-item predictions/losses, all aggregate tables, forecasts and failure/launch receipts. Three remote Codex login attempts failed before training; the master continued via a persistent ordinary-Python cluster supervisor.','',
    '**TLDR-end:** [the-difficulty-confound: 03_answer_matched_alignment] The complete matched-answer study reports both frozen gates and all counterevidence; claims are limited to this controlled addition setup.']
    (ROOT.parent/'RESULTS_matched_alignment.md').write_text('\n'.join(lines)+'\n')
    print('Report and figures generated from verified artifacts.',flush=True)

if __name__=='__main__':main()
