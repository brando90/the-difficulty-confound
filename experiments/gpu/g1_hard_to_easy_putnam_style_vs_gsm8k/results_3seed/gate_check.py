"""Gate + seed-0-vs-pilot parts of seeds3_report.py, run against a copy of the pilot runs with DONE markers
(results_pilot/runs/runs has no DONE files, so seeds3_report.py skipped both). Usage:
  python gate_check.py --pilot <dir with pilot run dirs + DONE> --out-root runs --json gate.json"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "seeds3"))
import numpy as np
import seeds3_report as S

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", required=True)
ap.add_argument("--out-root", required=True)
ap.add_argument("--json", required=True)
a = ap.parse_args()
S.PILOT = Path(a.pilot)
runs = Path(a.out_root) / "runs"
new = S.load(runs / S.BASE)
old = S.load(S.PILOT / S.BASE)
chk = {}
for w in ("gsm8k", "math500"):
    n = np.array(new[w]["correct"], float); o = np.array(old[w]["correct"], float)
    lo, hi = S.boot_ci(n - o); d = 100 * float((n - o).mean())
    chk[w] = {"pilot": round(100 * o.mean(), 2), "new": round(100 * n.mean(), 2), "delta": round(d, 2),
              "ci95_items": [round(lo, 2), round(hi, 2)], "item_agreement": round(100 * float((n == o).mean()), 2),
              "pass": lo <= 0 <= hi and abs(d) <= 1.5}
chk["pass"] = chk["gsm8k"]["pass"] and chk["math500"]["pass"]
s0 = {}
for arm in "EH":
    for lr in S.LRS:
        r = S.rid(arm, lr, 0); p = S.load(S.PILOT / r); g = S.load(runs / r)
        if p and g:
            s0[r] = {k: [round(100 * p[k]["acc"], 2), round(100 * g[k]["acc"], 2)] for k in ("gsm8k", "math500")}
json.dump({"grader_check": chk, "seed0_vs_pilot": s0}, open(a.json, "w"), indent=1)
print(json.dumps({"grader_check": chk, "seed0_vs_pilot": s0}, indent=1))
