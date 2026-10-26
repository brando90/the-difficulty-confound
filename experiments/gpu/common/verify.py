"""Check a plan's results.json against the raw run files, with code independent of analyze.py.

1. Integrity: every run the plan needs is DONE, and the SHA-256 in DONE matches metrics.jsonl.
2. Completeness: every checkpoint has every evaluation row it should have.
3. Recomputation: the headline numbers are recomputed in plain Python (no NumPy, no shared helpers)
   and must match results.json to 1e-9.
4. Grading: for generation rows, the share of items where Math-Verify and the fallback grader agree.

    python verify.py --plan g1 --tier main --out-root $OUT --report-dir ../g1_hard_to_easy_putnam_style_vs_gsm8k/results_main
    python verify.py --plan g3 --tier main --out-root $OUT --data-root $DATA --report-dir ...
Exit code 0 only if every check passes; prints VERIFY PASS or VERIFY FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import runs

TOL = 1e-9
BPB_SETS = ["L1_heldout", "L2_heldout", "L3_heldout", "gsm8k", "math500", "poetry_heldout", "web_heldout"]


def load(out_root: Path, rid: str) -> dict:
    d = runs.run_dir(out_root, rid)
    rows = {}
    for line in (d / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["step"], r["set"], r["kind"])] = r
    return rows


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def close(a: float, b: float) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, choices=runs.ALL_PLANS)
    ap.add_argument("--tier", default="main")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--secondary", action="store_true")
    ap.add_argument("--report-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_root)
    res = json.loads((Path(a.report_dir) / "results.json").read_text())
    specs = runs.expand(a.tier, [a.plan], a.secondary, allow_unset_lr=True)
    fails = []

    # 1-2 integrity and completeness
    data = {}
    for s in specs:
        d = runs.run_dir(out, s["run_id"])
        if not (d / "DONE").exists():
            fails.append(f"not DONE: {s['run_id']}")
            continue
        want = (d / "DONE").read_text().split()[0]
        if want != "fake" and hashlib.sha256((d / "metrics.jsonl").read_bytes()).hexdigest() != want:
            fails.append(f"hash mismatch: {s['run_id']}")
        rows = load(out, s["run_id"])
        data[s["run_id"]] = rows
        for st in s["ckpt_steps"]:
            for e in BPB_SETS:
                if (st, e, "bpb") not in rows:
                    fails.append(f"{s['run_id']}: missing bpb {e} at step {st}")
        for st in s["gen_steps"]:
            for e, k in (("gsm8k", "acc"), ("math500", "acc"), ("hellaswag", "mc")):
                if (st, e, k) not in rows:
                    fails.append(f"{s['run_id']}: missing {k} {e} at step {st}")
    if fails:
        print("\n".join(fails[:50]))
        print("VERIFY FAIL")
        return 1

    def arm(name):
        return [s["run_id"] for s in sorted(specs, key=lambda s: s["seed"]) if f"{a.plan}:{name}" in s["used_by"]]

    base = next(s["run_id"] for s in specs if s["kind"] == "base")
    F = next(s for s in specs if s["kind"] == "train")["max_steps"]

    def acc(rids, set_, mask=None):
        vals = []
        for r in rids:
            c = data[r][(F, set_, "acc")]["correct"]
            if mask is not None:
                c = [x for x, m in zip(c, mask) if m]
            vals.append(sum(c) / len(c))
        return mean(vals)

    def bpb(rid, step, set_):
        r = data[rid][(step, set_, "bpb")]
        return sum(p * b for p, b in zip(r["per_item"], r["bytes"])) / sum(r["bytes"])

    checks = []
    if a.plan == "g1":
        levels = data[arm("E")[0]][(F, "math500", "acc")]["levels"]
        hard = [lv is not None and lv >= 4 for lv in levels]
        checks.append(("A1", 100 * (acc(arm("E"), "gsm8k") - acc(arm("H"), "gsm8k")), res["gates"]["A1_easy_not_subsumed"]["estimate_pts"]))
        checks.append(("A2", 100 * (acc(arm("H"), "math500", hard) - acc(arm("E"), "math500", hard)), res["gates"]["A2_hard_end_needs_hard_data"]["estimate_pts"]))
        if "A3b_repeated_hard_slice_forgets" in res["gates"]:
            checks.append(("A3b", 100 * (acc(arm("H"), "gsm8k") - acc(arm("Hrep"), "gsm8k")), res["gates"]["A3b_repeated_hard_slice_forgets"]["estimate_pts"]))
    elif a.plan == "g2":
        b0 = bpb(base, 0, "poetry_heldout")
        # seed-mean of per-item bits, then ratio (as PLAN.md defines R)
        def seedmean_bpb(rids):
            rows = [data[r][(F, "poetry_heldout", "bpb")] for r in rids]
            byts = rows[0]["bytes"]
            bits = [mean(rw["per_item"][i] * byts[i] for rw in rows) for i in range(len(byts))]
            return sum(bits) / sum(byts)
        bp = seedmean_bpb(arm("P"))
        for k in ("E", "H", "N"):
            checks.append((f"R_{k}", (b0 - seedmean_bpb(arm(k))) / (b0 - bp), res["transfer_ratio"][k]["estimate"]))
    else:
        if not a.data_root:
            print("--data-root is needed for g3 (the complexity proxy K)")
            return 1
        cfg = json.loads((runs.PLAN_DIRS["g3"] / "config.json").read_text())
        man = json.loads((Path(a.data_root) / runs.load_suite()["data_version"] / "manifest.json").read_text())
        K = {k: float(v["median_tokens"]) for k, v in man["complexity_K"]["levels"].items()}
        t = next(s for s in specs if s["kind"] == "train")
        cs = t["ckpt_steps"]
        toks = [s * t["seqs_per_step"] * t["seq_len"] for s in cs]
        curves = {}
        for name, armc in cfg["arms"].items():
            if armc["distractor"] != "web":
                continue
            for seed, rid in enumerate(arm(name)):
                lv = armc["level"]
                curves[(lv, armc["alpha"], seed)] = [bpb(base, 0, f"{lv}_heldout") - bpb(rid, s, f"{lv}_heldout") for s in cs]
        dmax = {}
        for lv in cfg["levels"]:
            f = sorted(c[-1] for (l_, al, _), c in curves.items() if l_ == lv and al == 1.0)
            dmax[lv] = f[len(f) // 2] if len(f) % 2 else (f[len(f) // 2 - 1] + f[len(f) // 2]) / 2
        cell = {}
        for (lv, al, sd), d in curves.items():
            thr = 0.5 * dmax[lv]
            ts, ok = None, False
            if d[0] < thr:
                for i in range(1, len(d)):
                    if d[i] >= thr:
                        x0, x1 = math.log(toks[i - 1]), math.log(toks[i])
                        ts, ok = math.exp(x0 + (thr - d[i - 1]) / (d[i] - d[i - 1]) * (x1 - x0)), True
                        break
            cell.setdefault((lv, al), []).append(ts if ok else None)
        pts = [(math.log(K[lv]), math.log(al), mean(math.log(x) for x in v)) for (lv, al), v in cell.items() if all(x is not None for x in v)]
        # normal equations for y = c + eK*x1 - ea*x2, solved by Cramer's rule
        n = len(pts)
        sx1 = sum(p[0] for p in pts); sx2 = sum(-p[1] for p in pts); sy = sum(p[2] for p in pts)
        s11 = sum(p[0] ** 2 for p in pts); s22 = sum(p[1] ** 2 for p in pts); s12 = sum(-p[0] * p[1] for p in pts)
        s1y = sum(p[0] * p[2] for p in pts); s2y = sum(-p[1] * p[2] for p in pts)
        M = [[n, sx1, sx2], [sx1, s11, s12], [sx2, s12, s22]]
        v = [sy, s1y, s2y]

        def det(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        D = det(M)
        sol = []
        for j in range(3):
            Mj = [row[:] for row in M]
            for i in range(3):
                Mj[i][j] = v[i]
            sol.append(det(Mj) / D)
        checks.append(("e_K", sol[1], res["primary"]["pooled_fit"]["e_K"]))
        checks.append(("e_alpha", sol[2], res["primary"]["pooled_fit"]["e_alpha"]))
        checks.append(("E", sol[2] / sol[1], res["primary"]["pooled_fit"]["E"]))

    ok = True
    for name, mine, theirs in checks:
        good = close(float(mine), float(theirs))
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: recomputed {mine:.10g} vs results.json {theirs:.10g}")
    # 4 grader agreement
    ag = [r["grader_agreement"] for rows in data.values() for (st, e, k), r in rows.items() if k == "acc"]
    if ag:
        print(f"Math-Verify vs fallback grader agreement: mean {mean(ag):.3f}, min {min(ag):.3f} over {len(ag)} generation rows")
    print("VERIFY PASS" if ok else "VERIFY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
