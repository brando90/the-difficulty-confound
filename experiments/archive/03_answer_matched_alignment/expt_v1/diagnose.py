"""Post hoc failure diagnostics only; no refitting or changed forecasts."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze import read, design, axis
from addition import write_csv, save_json
ROOT=Path(__file__).resolve().parent
c=json.loads((ROOT/'config.json').read_text());out=ROOT/'results'
cal=read(out/'calibration_metrics.csv')
with (out/'scored_forecasts.csv').open() as f:scored=list(csv.DictReader(f))
fits=json.loads((out/'frozen_forecasts.json').read_text())['fits']
joint=[r for r in scored if r['axis']=='joint' and r['law']=='starting_difficulty']
total_sse=sum(float(r['error'])**2 for r in joint);rows=[]
for seed in c['seeds']:
    fit=next(f for f in fits if f['seed']==seed and f['law']=='starting_difficulty')
    train=[r for r in cal if r['seed']==seed and axis(r,c)=='fit']
    x,names=design(train,'starting_difficulty');z=x[:,1:];z=(z-z.mean(0))/z.std(0)
    ext=[r for r in cal if r['seed']==seed and axis(r,c)=='joint']
    own=[r for r in joint if int(float(r['seed']))==seed]
    base=[r for r in scored if r['axis']=='joint' and r['law']=='structure_exposure' and int(float(r['seed']))==seed]
    sse=sum(float(r['error'])**2 for r in own)
    rows.append(dict(seed=seed,design_rank=fit['rank'],raw_condition_number=float(np.linalg.cond(x)),standardized_condition_number=float(np.linalg.cond(np.column_stack([np.ones(len(z)),z]))),starting_difficulty_coefficient=fit['coefficients'][-1],fit_difficulty_min=min(r['starting_bits'] for r in train),fit_difficulty_max=max(r['starting_bits'] for r in train),joint_difficulty_min=min(r['starting_bits'] for r in ext),joint_difficulty_max=max(r['starting_bits'] for r in ext),joint_rmse=float(np.sqrt(sse/len(own))),baseline_joint_rmse=float(np.sqrt(np.mean([float(r['error'])**2 for r in base]))),joint_squared_error_share=sse/total_sse,max_joint_prediction=max(float(r['predicted_bits']) for r in own)))
write_csv(out/'posthoc_prediction_diagnostics.csv',rows)
worst=max(joint,key=lambda r:float(r['predicted_bits']))
summary=dict(label='POST HOC descriptive failure diagnostic; original gates unchanged',seeds=rows,largest_joint_forecast=worst)
save_json(out/'posthoc_prediction_diagnostics.json',summary)
print(json.dumps(summary,indent=2))
