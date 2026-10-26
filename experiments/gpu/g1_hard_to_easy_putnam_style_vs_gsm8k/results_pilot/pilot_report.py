#!/usr/bin/env python3
"""g1 pilot first-look report (stdlib only). Usage: python3 pilot_report.py OUT_ROOT [JOBIDS...]
Reads OUT_ROOT/runs/<run_id>/metrics.jsonl for the 7 g1 pilot runs, applies the pre-registered readout in CKPT.md."""
import json, math, random, subprocess, sys
from pathlib import Path
out = Path(sys.argv[1]); jobs = sys.argv[2:]
T = "T152x32x2048"; LRS = ["5e-06", "1e-05", "2e-05"]
def rid(pool, lr): return f"qwen2.5-0.5b__{pool}1__{T}__lr{lr}__s0"
def rows(r):
    m = out / "runs" / r / "metrics.jsonl"
    if not m.exists(): return None
    return [json.loads(x) for x in m.read_text().splitlines() if x.strip()]
def last(rs, s, kind):
    c = [x for x in rs if x["set"] == s and x["kind"] == kind]
    return max(c, key=lambda x: x["step"]) if c else None
def boot(a, b, n=10000, seed=0):
    d = [x - y for x, y in zip(a, b)]; N = len(d); rng = random.Random(seed); m = sum(d) / N
    bs = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return 100 * m, 100 * bs[int(0.025 * n)], 100 * bs[int(0.975 * n) - 1]
rep = {"runs": {}, "done": {}}
for name in [rid(p, lr) for p in ("L1_gradeschool", "L3_olympiad") for lr in LRS] + ["qwen2.5-0.5b__base"]:
    rep["done"][name] = (out / "runs" / name / "DONE").exists()
    rs = rows(name)
    if not rs: continue
    g = last(rs, "gsm8k", "acc"); m5 = last(rs, "math500", "acc")
    e = {"gsm8k": g and round(100 * g["acc"], 2), "math500": m5 and round(100 * m5["acc"], 2)}
    if m5:
        hi = [c for c, l in zip(m5["correct"], m5["levels"]) if str(l) in ("4", "5", "Level 4", "Level 5")]
        e["math500_L45"] = round(100 * sum(hi) / max(len(hi), 1), 2); e["n_L45"] = len(hi)
    for s in ("L1_heldout", "L3_heldout", "web_heldout"):
        b = last(rs, s, "bpb"); e[s + "_bpb"] = b and round(b["bpb"], 4)
    hs = last(rs, "hellaswag", "mc"); e["hellaswag"] = hs and round(100 * hs.get("acc_norm", hs.get("acc", 0)), 2)
    sp = out / "runs" / name / "summary.json"
    if sp.exists():
        sj = json.loads(sp.read_text()); e.update({k: sj.get(k) for k in ("tokens_per_second", "train_seconds", "wall_seconds", "peak_gpu_gb")})
    rep["runs"][name] = e
# select-lr rule (same as runs.py select-lr, without writing)
sc = {}
for lr in LRS:
    try:
        v = [last(rows(rid("L1_gradeschool", lr)), "L1_heldout", "bpb")["bpb"], last(rows(rid("L3_olympiad", lr)), "L3_heldout", "bpb")["bpb"]]
        sc[lr] = sum(v) / 2 if all(math.isfinite(x) for x in v) else float("inf")
    except Exception: pass
if sc:
    best = min(sc.values()); sel = min((lr for lr in sc if sc[lr] <= best + 0.002), key=float); rep["selected_lr"] = sel; rep["lr_scores"] = sc
rep["readout"] = {}
for lr in LRS:
    E, H = rows(rid("L1_gradeschool", lr)), rows(rid("L3_olympiad", lr))
    if not E or not H: continue
    gE, gH = last(E, "gsm8k", "acc"), last(H, "gsm8k", "acc"); r = {}
    if gE and gH: r["gsm8k_E_minus_H"] = [round(x, 2) for x in boot(gE["correct"], gH["correct"])]
    mE, mH = last(E, "math500", "acc"), last(H, "math500", "acc")
    if mE and mH:
        idx = [i for i, l in enumerate(mE["levels"]) if str(l) in ("4", "5", "Level 4", "Level 5")]
        if idx: r["math500L45_H_minus_E"] = [round(x, 2) for x in boot([mH["correct"][i] for i in idx], [mE["correct"][i] for i in idx])]
    b = rows("qwen2.5-0.5b__base")
    gb = b and last(b, "gsm8k", "acc")
    if gb and gH: r["gsm8k_H_minus_base"] = [round(x, 2) for x in boot(gH["correct"], gb["correct"])]
    rep["readout"][lr] = r
if jobs:
    try:
        rep["sacct"] = subprocess.run(["sacct", "-X", "-j", ",".join(jobs), "--format=JobID%20,JobName%14,Partition,State,Elapsed,NodeList,AllocTRES%60"], capture_output=True, text=True).stdout
    except Exception as ex: rep["sacct"] = str(ex)
print(json.dumps(rep, indent=1))
