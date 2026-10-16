"""Render the frozen analysis without choosing experiments or fitting new models."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
HOME=ROOT.parent


def read(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    runtime=json.loads((OUT/'runtime.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    metrics=read('forecast_metrics.csv');comparisons=read('forecast_comparisons.csv');seeds=read('seed_forecast_metrics.csv')
    scores=read('scored_forecasts.csv')
    primary=summary['primary']
    names=['resources','additive','exposure','difficulty','full','structural','full_structural']
    labels=['Resources','Additive baseline','+ Log exposure','+ Starting difficulty','+ Both covariates','+ Carry categories','+ Both and carry']
    colors=['#8d98a5','#335c81','#84a6bf','#82b7ab','#007d73','#b5a0c4','#60528a']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':200})
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),gridspec_kw={'width_ratios':[1.05,1,1]})
    selected=[next(r for r in metrics if r['axis']=='joint' and r['law']==name) for name in names]
    vals=np.array([float(r['rmse']) for r in selected])
    lo=np.array([float(r['rmse_lo']) for r in selected]);hi=np.array([float(r['rmse_hi']) for r in selected])
    axes[0].barh(np.arange(7),vals,color=colors,xerr=np.array([vals-lo,hi-vals]),capsize=2)
    axes[0].set_yticks(np.arange(7),labels);axes[0].invert_yaxis();axes[0].set_xlabel('Forecast root mean squared error (bits/answer)');axes[0].set_title('Joint holdout: all frozen models')
    seed_ids=sorted({int(float(r['seed'])) for r in seeds})
    for i,width in enumerate([96,256]):
        gain=[]
        for seed in seed_ids:
            rr={r['law']:float(r['rmse']) for r in seeds if r['axis']==f'joint_width_{width}' and int(float(r['seed']))==seed}
            gain.append(100*(1-rr['full']/rr['additive']))
        axes[1].scatter(np.array(gain),np.arange(8)+i*.13,s=35,color=['#335c81','#007d73'][i],label=f'Width {width}')
    axes[1].axvline(0,color='#555',linestyle='--',linewidth=.8);axes[1].set_yticks(np.arange(8),[str(s) for s in seed_ids]);axes[1].set_xlabel('Error reduction with both covariates (%)');axes[1].set_ylabel('Independent seed block');axes[1].set_title('Both unseen widths, every seed');axes[1].legend(frameon=False)
    low=0;high=0
    for name,color,label in [('additive','#335c81','Additive baseline'),('full','#007d73','Both covariates')]:
        rr=[r for r in scores if r['axis']=='joint' and r['law']==name]
        x=np.array([float(r['observed_bits']) for r in rr]);y=np.array([float(r['predicted_bits']) for r in rr])
        axes[2].scatter(x,y,s=10,alpha=.3,color=color,label=label)
        low=min(low,float(y.min()));high=max(high,float(max(x.max(),y.max())))
    axes[2].plot([low,high],[low,high],color='#333',linewidth=.8,linestyle='--');axes[2].set_xlabel('Observed test loss (bits/answer)');axes[2].set_ylabel('Forecast loss (bits/answer)');axes[2].set_title('384 joint held-out aggregate targets');axes[2].legend(frameon=False)
    fig.suptitle('Fixed three-digit addition: held-out model widths and exposure mixtures',fontsize=14)
    fig.text(.5,.015,'Intervals: 95% paired bootstrap over 8 seed blocks; 20,000 draws. Test questions were excluded from every fit.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.95])
    for suffix in ['png','pdf','svg']:
        fig.savefig(HOME/f'heldout_laws.{suffix}',bbox_inches='tight')
    plt.close(fig)
    lines=['# Results: held-out exposure and difficulty laws','','**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/02_heldout_exposure_difficulty_laws/RESULTS_heldout_laws.md>','',f"**TLDR:** {summary['verdict']}: adding log exposure and independently probed starting difficulty changed joint held-out forecast error by a reduction of {100*primary['relative_rmse_gain']:.2f}% [{100*primary['ci95_lo']:.2f}, {100*primary['ci95_hi']:.2f}]. This is a controlled transfer test of the alignment-law functional form, not validation on Varun's original language models.",'',f"Completed on 10-16-2026: {runtime['completed_cells']}/200 intervention trajectories and 40 warm starts, eight independent seed blocks, {runtime['elapsed_seconds']/60:.2f} training minutes on skampere2, two processor threads. No language-model provider calls. Frozen source: `{runtime['git_revision']}`. [Protocol](PROTOCOL.md) · [Configuration](expt_v1/config.json) · [Verification](expt_v1/results/verification.json).",'', '![Held-out predictive-law comparison](heldout_laws.png)','', '## What was predicted','', 'The outcome is mean correct-answer negative log probability in bits for each maximum carry-chain stratum. Every trained network performs the same exact three-digit addition task. We fit the law on widths 32, 64, 128 and long-carry training fractions .02, .25, .80; withheld widths 96 and 256 and fractions .10 and .50 never contribute outcomes to fitting. The parameter counts are 4,328 (width 32), 10,664 (64), 29,480 (128), 19,048 (96), and 91,688 (256). The two withheld mixtures interpolate between fitting mixtures; width 96 interpolates and width 256 extrapolates in capacity. All five mixtures at a given width and seed share the same balanced 500-update warm start and optimizer reset.', '', 'Each seed has 108 calibration aggregates. The primary joint holdout has 48 targets per seed, 384 total, at 300, 1,000 and 3,000 updates and four carry strata. Targets weight each width, mixture, checkpoint, and carry stratum equally; this is not weighting by the natural frequency of carry patterns. These aggregates share 4,000 test questions, not 384 independent training replicates. One thousand separate questions calibrate the laws; another thousand probe starting difficulty. Every training pool excludes all 6,000 held-out questions and their operand swaps. Training examples can overlap across mixture conditions, deliberately preserving the paired comparison.', '', 'The transferred baseline is loss = a + b N^(-beta) + c T^(-gamma) - d q. Here N is parameter count, T is intervention updates, and q is the known training fraction of the evaluated stratum. Replacing the original semantic alignment score with q is an explicit adaptation. The full extension adds e log(q) + f D, where D is the starting model\'s mean loss on its independent probe. The log-exposure term is a nonlinear transformation of q, which the baseline already observes; it adds functional flexibility, not new information. D supplies the new measurement. Obtaining D requires 1,000 probe evaluations per starting model, including each unseen width. It never uses a future outcome. This is supervised neural-network training, not reinforcement learning.', '', '## Primary held-out result','', '| Forecast model | Root mean squared error, 95% interval | Mean absolute error | Signed bias | Prediction mean | Outcome mean | Negative forecasts / 384 |','|---|---:|---:|---:|---:|---:|---:|']
    for label,r in zip(labels,selected):
        lines.append(f"| {label} | {float(r['rmse']):.4f} [{float(r['rmse_lo']):.4f}, {float(r['rmse_hi']):.4f}] | {float(r['mae']):.4f} | {float(r['bias']):+.4f} | {float(r['prediction_mean']):.4f} | {float(r['outcome_mean']):.4f} | {r['negative_predictions']} |")
    lines += ['', 'Units are bits per answer. Signed bias is prediction minus observation; positive means pessimistic forecasts of loss. Intervals are paired percentile bootstrap intervals over eight independently fitted seed blocks (20,000 draws). They condition on this fixed question set and model family; checkpoints and questions are not resampled as independent runs. p-val=n/a: this is an estimation comparison, not a null-hypothesis test.', '', 'The prospective gate requires at least a 10% joint error reduction, positive lower confidence bound, and positive point improvement at both held widths. The 10% practical threshold was chosen in our protocol. The verdict is computed from all three conditions, without selecting a checkpoint or seed.', '', '| Holdout | Full versus additive: relative error reduction, 95% interval | Target aggregates |','|---|---:|---:|']
    for axis in ['joint','joint_width_96','joint_width_256','model_only','mixture_only']:
        r=next(r for r in comparisons if r['axis']==axis and r['baseline']=='additive' and r['augmented']=='full')
        n=next(r['n_targets'] for r in metrics if r['axis']==axis and r['law']=='full')
        lines.append(f"| {axis} | {100*float(r['relative_rmse_gain']):.2f}% [{100*float(r['ci95_lo']):.2f}, {100*float(r['ci95_hi']):.2f}] | {n} |")
    lines += ['', '## Do both covariates help?','', 'A full-model improvement alone cannot establish an independent contribution from each covariate. These secondary ablations keep the same fitting and joint holdout protocol; they are descriptive, with no multiple-comparison significance claims.', '', '| Baseline → augmented | Relative error reduction, 95% interval |','|---|---:|']
    for r in comparisons:
        if r['axis']=='joint' and not(r['baseline']=='additive' and r['augmented']=='full'):
            lines.append(f"| {r['baseline']} → {r['augmented']} | {100*float(r['relative_rmse_gain']):.2f}% [{100*float(r['ci95_lo']):.2f}, {100*float(r['ci95_hi']):.2f}] |")
    lines += ['', '## Interpretation and limits','', '- The prediction task, test identities, notation, and evaluation rule are fixed. Width is deliberately varied; conclusions are about a fixed operational task, not proof that latent internal strategies are identical.', '- Exposure changes the training distribution and also digit/answer marginals. This identifies the specified distribution intervention, not a uniquely carry-specific mechanism. More diverse language tasks and additional model families remain untested.', '- D is a measured property of a starting model on independent questions. It is not an intrinsic, model-independent property of an item, nor an estimate of unknown pretraining frequency. It can capture capacity, optimization, and structural effects as well as learned familiarity; its predictive value alone does not identify exposure as the dominant cause.', '- The seven models, exponent grid, coefficient bounds, and decisions were frozen before outcomes. The nonlinear exponent grid is coarse; these are declared adaptations, not the original continuous optimizer. Forecasts are not clipped; negative loss predictions reveal limitations of the additive form.', '- Conditional predictive code length measures model-dependent uncertainty. Formal conditional Kolmogorov complexity stays fixed for unchanged input, answer, specification, and universal machine; it is not computable in general. The measured bits exclude the costs of describing the model and data, so they are not a complete minimum-description-length result.', '- A positive result supports the narrow claim that known exposure and an independent starting-model probe help this particular forecasting law on these held-out conditions. It cannot establish that difficulty is mostly data across language models.', '', '## Actual alignment-law observations','', 'A companion retrospective analysis was executed on 24 mathematics observations from the original, private alignment repository. Available metadata support a training-breadth proxy; independently measured pre-update difficulty and repeated exposure are absent. Its inputs and derived outputs are retained privately pending publication permission. No original-data claim is substituted by the controlled addition result.', '', '## Verification and reproduction','', f"Independent verification passed: {verification['per_item_prediction_rows_verified']:,} stored per-item predictions, {verification['aggregate_outcomes_verified']:,} aggregates, {verification['law_forecasts_verified']:,} forecasts, all 40 training manifests, all {verification['weights_replayed']} warm/final weight files, zero split overlap, and independently recomputed primary bootstrap. Final weight replay has maximum absolute probability-code-length difference {verification['max_cross_platform_bits_delta']:.3g} bits on this verification host.", '', 'The committed result tables and compressed NumPy per-item arrays are sufficient to reproduce every public number without cluster access. Frozen coefficients and forecasts are saved before the first final-test outcome read in the analysis process. Source, calibration, configuration, and forecast hashes are recorded. Training wrote the final-test files earlier, but the fitting process did not read them before freezing forecasts.', '', '```bash','python experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/verify_artifacts.py','python experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/report.py','```','', 'The verifier checks recorded forecasts against frozen coefficients and recomputes the primary interval. Refitting with `analyze.py` requires the recorded numerical environment and refuses to overwrite changed frozen forecasts. A fresh training reproduction must use a new output path: `run.py --output /path/to/new-run`, followed by `analyze.py --output /path/to/new-run`.', '', 'Detailed tables: [all metrics](expt_v1/results/forecast_metrics.csv), [all ablations](expt_v1/results/forecast_comparisons.csv), [every seed](expt_v1/results/seed_forecast_metrics.csv), [all scored forecasts](expt_v1/results/scored_forecasts.csv), [frozen coefficients](expt_v1/results/frozen_forecasts.json).']
    (HOME/'RESULTS_heldout_laws.md').write_text('\n'.join(lines)+'\n')
    print('Report and PNG/PDF/SVG figures written.')

if __name__=='__main__':
    main()
