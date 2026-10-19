"""Write fake DONE runs with planted effects, to check that the three analyses recover known answers.

Never a result: the output folder is marked FAKE and analyses read it only through this test.

    python fake_runs.py --out-root /tmp/fake --tier main
    (then run each plan's analyze.py and verify.py on /tmp/fake; smoke_test.sh does all of this)

Planted truth: g1 GSM8K E - H = +7 points, MATH-500 L4-5 H - E = +6 points, H loses 3 GSM8K points
between its best earlier checkpoint and the end, H - Hrep = +6 points; g2 R_H = -0.06, R_E = -0.04, R_N = +0.10;
g3 T* = c K^1 alpha^-0.8, so e_K = 1, e_alpha = 0.8 and E = 0.8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import runs

N_ITEMS = {"gsm8k": 1319, "math500": 500, "L1_heldout": 1000, "L2_heldout": 1000, "L3_heldout": 1000,
           "poetry_heldout": 1000, "web_heldout": 1000, "hellaswag": 2000}
FAKE_K = {"L1": 150.0, "L2": 300.0, "L3": 600.0}
BASE_BPB = {"L1_heldout": 0.9, "L2_heldout": 1.0, "L3_heldout": 1.1, "gsm8k": 0.9, "math500": 1.0, "poetry_heldout": 2.5, "web_heldout": 2.0}
GSM = {"base": .30, "E": .45, "H": .38, "M": .44, "N": .30, "Hrep": .32, "P": .30}
M45 = {"base": .05, "E": .08, "H": .14, "M": .13, "N": .05, "Hrep": .10, "P": .05}
POETRY_FINAL = {"E": 2.52, "H": 2.53, "N": 2.45, "P": 2.00}
ETA, KAPPA = 0.8, 1.0


def arm_of(spec: dict) -> str:
    tags = [u.split(":")[1] for u in spec["used_by"]]
    for k in ("Hrep", "M", "P", "N", "E", "H"):
        if k in tags:
            return k
    return "g3"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data-root", required=True, help="a fake manifest with K values is written here")
    ap.add_argument("--tier", default="main")
    a = ap.parse_args()
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "FAKE").write_text("planted-effect test data, never a result\n")
    dv = Path(a.data_root) / runs.load_suite()["data_version"]
    dv.mkdir(parents=True, exist_ok=True)
    (dv / "manifest.json").write_text(json.dumps({"toy": True, "complexity_K": {"proxy": "fake", "levels": {k: {"median_tokens": v} for k, v in FAKE_K.items()}}}))

    rng = np.random.default_rng(0)
    u = {s: rng.random(n) for s, n in N_ITEMS.items()}  # item difficulty, shared by all runs
    item_bpb = {s: BASE_BPB[s] + 0.3 * rng.standard_normal(N_ITEMS[s]) for s in BASE_BPB}
    nbytes = {s: rng.integers(200, 800, N_ITEMS[s]) for s in BASE_BPB}
    levels = np.array([1 + (i % 5) for i in range(500)])
    hard = levels >= 4

    specs = runs.expand(a.tier, runs.ALL_PLANS, secondary=True, allow_unset_lr=True)
    tps = next(s for s in specs if s["kind"] == "train")
    tps = tps["seqs_per_step"] * tps["seq_len"]
    F = next(s for s in specs if s["kind"] == "train")["max_steps"]
    for spec in specs:
        r = np.random.default_rng(int(hashlib.sha256(spec["run_id"].encode()).hexdigest()[:8], 16))
        arm = "base" if spec["kind"] == "base" else arm_of(spec)
        mix = spec.get("mixture", {})
        rows = []
        for step in spec["ckpt_steps"]:
            T = step * tps
            frac = 1.0 if spec["kind"] == "base" else min(1.0, math.log(1 + T / 1e6) / math.log(1 + F * tps / 1e6))
            for s in BASE_BPB:
                d = 0.0
                if s == "poetry_heldout" and arm in POETRY_FINAL:
                    d = (BASE_BPB[s] - POETRY_FINAL[arm]) * frac
                for lvl, pool in (("L1", "L1_gradeschool"), ("L2", "L2_competition"), ("L3", "L3_olympiad")):
                    if s == f"{lvl}_heldout" and mix.get(pool, 0) > 0 and spec["kind"] == "train":
                        alpha = mix[pool]
                        c = 1e6 / FAKE_K["L1"] * math.exp(0.05 * r.standard_normal())
                        tstar = c * FAKE_K[lvl] ** KAPPA * alpha ** (-ETA)
                        d = 0.4 * (1 - math.exp(-math.log(2) * T / tstar))
                per = item_bpb[s] - d + 0.01 * r.standard_normal(N_ITEMS[s])
                b = nbytes[s]
                rows.append({"step": step, "tokens": T, "set": s, "kind": "bpb", "bpb": float((per * b).sum() / b.sum()),
                             "n": N_ITEMS[s], "per_item": [round(float(x), 6) for x in per], "bytes": [int(x) for x in b]})
            if step in spec["gen_steps"]:
                p_g = GSM.get(arm, .30)
                p_m = M45.get(arm, .05)
                if arm == "H":  # planted forgetting: best at the middle generation checkpoint, 3 points lower at the end
                    p_g = {40: .40, 160: .41}.get(step, .38)
                elif spec["kind"] == "train" and step < F:
                    p_g = GSM["base"] + (p_g - GSM["base"]) * 0.8
                noise = 0.03 * r.standard_normal(N_ITEMS["gsm8k"])
                cg = (u["gsm8k"] + noise < p_g).astype(int)
                um = u["math500"] + 0.03 * r.standard_normal(500)
                cm = np.where(hard, um < p_m, um < p_m + 0.3).astype(int)
                for s, c in (("gsm8k", cg), ("math500", cm)):
                    rows.append({"step": step, "tokens": T, "set": s, "kind": "acc", "acc": float(c.mean()), "acc_fallback": float(c.mean()),
                                 "grader_agreement": 1.0, "n": len(c), "correct": [int(x) for x in c],
                                 "levels": [int(x) for x in levels] if s == "math500" else [None] * len(c)})
                ch = (u["hellaswag"] + 0.05 * r.standard_normal(2000) < 0.4).astype(int)
                rows.append({"step": step, "tokens": T, "set": "hellaswag", "kind": "mc", "acc_norm": float(ch.mean()), "n": 2000, "correct": [int(x) for x in ch]})
        d = runs.run_dir(out, spec["run_id"])
        d.mkdir(parents=True, exist_ok=True)
        with (d / "metrics.jsonl").open("w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        (d / "spec.json").write_text(json.dumps(spec))
        (d / "DONE").write_text("fake\n")
    print(f"wrote {len(specs)} fake runs to {out}")


if __name__ == "__main__":
    main()
