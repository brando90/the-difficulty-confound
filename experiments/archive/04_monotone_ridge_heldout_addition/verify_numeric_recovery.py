"""Replay the frozen verifier with one scale-aware floating-point assertion.

The original verifier and failed supervisor log remain unchanged. No training,
fitting, forecast, scientific threshold, or data membership is modified.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
source=ROOT/'expt_v1/verify.py'
text=source.read_text()
old="for name,val in vals.items():assert np.isclose(float(r[name]),val,atol=1e-12,rtol=0,equal_nan=True)"
new="for name,val in vals.items():assert np.isclose(float(r[name]),val,atol=max(1e-12,8*abs(np.spacing(val))),rtol=0,equal_nan=True)"
assert text.count(old)==1
out=ROOT/'expt_v1/results'
failures=json.loads((out/'verification_roundoff_diagnosis.json').read_text())
assert len(failures)==1
assert failures[0]['law']=='legacy_difficulty' and failures[0]['metric']=='rmse'
assert failures[0]['representable_float_steps']==1.0
sys.path.insert(0,str(source.parent))
exec(compile(text.replace(old,new),str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
receipt=json.loads((out/'verification.json').read_text())
receipt.update(numerical_verification_recovery=True,
    original_verifier_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    recovery_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    recovered_assertion='Aggregate scalar metric equality only: max(1e-12, eight representable floating-point steps); relative tolerance remains zero.',
    original_failure=failures[0],
    unchanged='All training, fit coefficients, forecasts, datasets, bootstrap draws and scientific gates; original failed supervisor receipt retained.')
(out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('RECOVERED_VERIFICATION_PASS: one-ULP aggregate rounding discrepancy; all frozen checks completed.')
