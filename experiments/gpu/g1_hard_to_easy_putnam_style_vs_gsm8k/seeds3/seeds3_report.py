#!/usr/bin/env python3
"""g1 3-seed check (tier seeds3): the readout pre-registered in experiments/gpu/CKPT.md, from the per-item files.

    python seeds3_report.py --out-root $OUT --json g1s3_report.json --md g1s3_report.md

Reads OUT/runs/<run_id>/metrics.jsonl (per-item correctness at step 152) and the pilot's base-model per-item results
in results_pilot/ (the grader-equivalence check). Needs numpy. Every number it prints is recomputed from those files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G1 = HERE.parent
PILOT = G1 / "results_pilot" / "runs" / "runs"
T = "T152x32x2048"
SEEDS = [0, 1, 2]
LRS = ["5e-06", "2e-05"]
F_LR = "5e-06"
N_BOOT, BOOT_SEED = 10_000, 0
T975_DF2 = 4.302652729911275  # two-sided 95% Student t, 2 degrees of freedom


def rid(arm: str, lr: str, seed: int) -> str:
    mix = {"E": "L1_gradeschool1", "H": "L3_olympiad1", "F": "L1_gradeschool0.5+L3_olympiad0.5[answer_only]"}[arm]
    return f"qwen2.5-0.5b__{mix}__{T}__lr{lr}__s{seed}"


BASE = "qwen2.5-0.5b__base"


REGRADE = None  # --regrade pilot: GSM8K and MATH-500 from <run>/regrade_pilot/ (regrade.py), the rest from the run


def load(run_dir: Path) -> dict | None:
    m = run_dir / "metrics.jsonl"
    if not (run_dir / "DONE").exists() or not m.exists():
        return None
    rows = [json.loads(x) for x in m.read_text().splitlines() if x.strip()]
    if REGRADE and run_dir.parent.parent != PILOT.parent:
        rg = run_dir / f"regrade_{REGRADE}"
        if not (rg / "DONE").exists():
            return None
        rows = [r for r in rows if r["set"] not in ("gsm8k", "math500") or r["kind"] != "acc"]
        rows += [json.loads(x) for x in (rg / "metrics.jsonl").read_text().splitlines() if x.strip()]
    out = {}
    for name, kind in (("gsm8k", "acc"), ("math500", "acc"), ("hellaswag", "mc"), ("L1_heldout", "bpb"), ("L3_heldout", "bpb")):
        c = [r for r in rows if r["set"] == name and r["kind"] == kind]
        if c:
            out[name] = max(c, key=lambda r: r["step"])
    return out


def hard(levels: list) -> np.ndarray:
    return np.array([str(lv) in ("4", "5", "Level 4", "Level 5") for lv in levels])


def vec(run: dict, which: str) -> np.ndarray:
    """Per-item correctness: 'gsm8k', or 'math500_L45' (MATH-500 levels 4-5 only)."""
    if which == "gsm8k":
        return np.array(run["gsm8k"]["correct"], dtype=float)
    m = run["math500"]
    return np.array(m["correct"], dtype=float)[hard(m["levels"])]


def boot_ci(d: np.ndarray) -> tuple[float, float]:
    """95% percentile interval of the mean of per-item values d, 10,000 item resamples, seed 0 (points)."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return 100 * float(np.quantile(means, 0.025)), 100 * float(np.quantile(means, 0.975))


def contrast(a_runs: list[dict], b_runs: list[dict] | None, which: str, b_single: dict | None = None) -> dict:
    """a - b. a_runs (and b_runs) are per-seed runs in seed order; b_single is one run (base) used for every seed.
    Per-seed differences, their mean, the across-seed t interval (secondary) and the primary interval: paired
    bootstrap over items of the seed-averaged per-item difference."""
    a = [vec(r, which) for r in a_runs]
    b = [vec(r, which) for r in b_runs] if b_runs else [vec(b_single, which)] * len(a)
    per_seed = [100 * float((x - y).mean()) for x, y in zip(a, b)]
    d = np.mean([x - y for x, y in zip(a, b)], axis=0)
    mean = 100 * float(d.mean())
    lo, hi = boot_ci(d)
    out = {"per_seed": [round(x, 2) for x in per_seed], "mean": round(mean, 2), "ci95_items": [round(lo, 2), round(hi, 2)],
           "n_items": int(len(d)), "n_seeds": len(a)}
    if len(per_seed) >= 2:
        sd = float(np.std(per_seed, ddof=1))
        half = (T975_DF2 if len(per_seed) == 3 else float("nan")) * sd / math.sqrt(len(per_seed))
        out["ci95_seeds_t"] = [round(mean - half, 2), round(mean + half, 2)]
    out["excludes_0"] = lo > 0 or hi < 0
    out["all_seeds_positive"] = all(x > 0 for x in per_seed)
    out["all_seeds_negative"] = all(x < 0 for x in per_seed)
    return out


def interaction(e2, h2, e5, h5, which: str) -> dict:
    """(E - H) at 2e-5 minus (E - H) at 5e-6, seed-averaged per item, paired bootstrap over items."""
    m = lambda rs: np.mean([vec(r, which) for r in rs], axis=0)  # noqa: E731
    d = (m(e2) - m(h2)) - (m(e5) - m(h5))
    lo, hi = boot_ci(d)
    return {"mean": round(100 * float(d.mean()), 2), "ci95_items": [round(lo, 2), round(hi, 2)], "excludes_0": lo > 0 or hi < 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--regrade", choices=["pilot"], help="read the re-graded generation results (after a failed grader check)")
    a = ap.parse_args()
    global REGRADE
    REGRADE = a.regrade
    runs_dir = Path(a.out_root) / "runs"
    want = [rid(arm, lr, s) for arm in "EH" for lr in LRS for s in SEEDS] + [rid("F", F_LR, s) for s in SEEDS] + [BASE]
    got = {r: load(runs_dir / r) for r in want}
    rep: dict = {"done": {r: got[r] is not None for r in want}, "n_done": sum(v is not None for v in got.values()), "n_runs": len(want)}

    # per-run table: accuracy, format diagnostics, held-out bits per byte, cost
    table = {}
    for r, m in got.items():
        if m is None:
            continue
        e = {}
        for name in ("gsm8k", "math500"):
            x = m[name]
            e[name] = round(100 * x["acc"], 2)
            for k in ("stopped_rate", "boxed_rate"):
                if k in x:
                    e[f"{name}_{k}"] = round(100 * x[k], 1)
            if "mean_tokens" in x:
                e[f"{name}_mean_tokens"] = round(x["mean_tokens"], 1)
            e[f"{name}_gen_seconds"] = x.get("seconds")
        e["math500_L45"] = round(100 * float(vec(m, "math500_L45").mean()), 2)
        e["hellaswag"] = round(100 * m["hellaswag"]["acc_norm"], 2) if "hellaswag" in m else None
        for s in ("L1_heldout", "L3_heldout"):
            e[f"{s}_bpb"] = round(m[s]["bpb"], 4) if s in m else None
        sj = runs_dir / r / "summary.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            e.update({k: s.get(k) for k in ("gpu", "train_seconds", "wall_seconds", "dropped_by_transform")})
        table[r] = e
    rep["runs"] = table
    rep["gpu_hours"] = round(sum((v.get("wall_seconds") or 0) for v in table.values()) / 3600, 3)

    # grader-equivalence check (pre-registered gate): new base vs the pilot's base, same items
    pb = load(PILOT / BASE)
    if got[BASE] and pb:
        chk = {}
        for which, name in (("gsm8k", "gsm8k"), ("math500", "math500")):
            new = np.array(got[BASE][name]["correct"], dtype=float)
            old = np.array(pb[name]["correct"], dtype=float)
            lo, hi = boot_ci(new - old)
            delta = 100 * float((new - old).mean())
            chk[which] = {"pilot": round(100 * float(old.mean()), 2), "new": round(100 * float(new.mean()), 2),
                          "delta": round(delta, 2), "ci95_items": [round(lo, 2), round(hi, 2)],
                          "item_agreement": round(100 * float((new == old).mean()), 2),
                          "pass": lo <= 0 <= hi and abs(delta) <= 1.5}
        chk["pass"] = chk["gsm8k"]["pass"] and chk["math500"]["pass"]
        rep["grader_check"] = chk

    # seed 0 of E and H against the pilot's seed-0 runs (descriptive: training is not bit-reproducible on GPU)
    rep["seed0_vs_pilot"] = {}
    for arm in "EH":
        for lr in LRS:
            r = rid(arm, lr, 0)
            p = load(PILOT / r)
            if got[r] and p:
                rep["seed0_vs_pilot"][r] = {"gsm8k": [round(100 * p["gsm8k"]["acc"], 2), round(100 * got[r]["gsm8k"]["acc"], 2)],
                                            "math500": [round(100 * p["math500"]["acc"], 2), round(100 * got[r]["math500"]["acc"], 2)]}

    # the pre-registered readout; computed only when every run it needs is DONE
    def runs_of(arm, lr):
        rs = [got[rid(arm, lr, s)] for s in SEEDS]
        return rs if all(rs) else None

    R: dict = {}
    for lr in LRS:
        E, H = runs_of("E", lr), runs_of("H", lr)
        if E and H:
            R[f"gsm8k_E_minus_H@{lr}"] = contrast(E, H, "gsm8k")
            R[f"math500L45_H_minus_E@{lr}"] = contrast(H, E, "math500_L45")
    F = runs_of("F", F_LR)
    for arm in "EH":
        for lr in LRS:
            X = runs_of(arm, lr)
            if X and F:
                for which in ("gsm8k", "math500_L45"):
                    c = contrast(X, F, which)
                    c["same_lr_as_F"] = lr == F_LR
                    R[f"{which}_{arm}@{lr}_minus_F"] = c
            if X and got[BASE]:
                R[f"gsm8k_{arm}@{lr}_minus_base"] = contrast(X, None, "gsm8k", b_single=got[BASE])
    if F and got[BASE]:
        R["gsm8k_F_minus_base"] = contrast(F, None, "gsm8k", b_single=got[BASE])
        R["math500L45_F_minus_base"] = contrast(F, None, "math500_L45", b_single=got[BASE])
    E2, H2, E5, H5 = runs_of("E", "2e-05"), runs_of("H", "2e-05"), runs_of("E", "5e-06"), runs_of("H", "5e-06")
    if E2 and H2 and E5 and H5:
        R["gsm8k_interaction_lr"] = interaction(E2, H2, E5, H5, "gsm8k")
    rep["readout"] = R

    # verdicts, word for word the rules in CKPT.md
    V = {}
    lo_c, hi_c = R.get("gsm8k_E_minus_H@5e-06"), R.get("gsm8k_E_minus_H@2e-05")
    if lo_c and hi_c and "gsm8k_interaction_lr" in R:
        neg = lo_c["mean"] < 0 and lo_c["excludes_0"] and lo_c["all_seeds_negative"]
        pos = hi_c["mean"] > 0 and hi_c["excludes_0"] and hi_c["all_seeds_positive"]
        inter = R["gsm8k_interaction_lr"]["excludes_0"] and R["gsm8k_interaction_lr"]["mean"] > 0
        V["sign_flip"] = ("REAL" if neg and pos and inter else
                          "LR-DEPENDENT, SIGN NOT ESTABLISHED" if inter else "NOT REAL AT THIS SCALE")
        V["P1_E_minus_H_ge_3_at_5e-6"] = "SUPPORTED" if lo_c["mean"] >= 3 and lo_c["ci95_items"][0] > 0 and lo_c["all_seeds_positive"] else "NOT SUPPORTED"
    hf = R.get("gsm8k_H@5e-06_minus_F")
    if hf:
        V["H_beats_format_only_on_gsm8k_at_5e-6"] = "YES" if hf["excludes_0"] and hf["mean"] > 0 and hf["all_seeds_positive"] else "NO"
    fb, hb = R.get("gsm8k_F_minus_base"), R.get("gsm8k_H@5e-06_minus_base")
    if fb and hb and hb["mean"] > 0:
        V["format_share_of_H_gain"] = round(fb["mean"] / hb["mean"], 3)
    if F:  # the CKPT.md caveat: F is read as a format control only if it still works problems out
        mt = [r["gsm8k"].get("mean_tokens") for r in F]
        V["F_answers_without_working"] = all(x is not None and x < 30 for x in mt)
    if "grader_check" in rep:
        V["grader_check"] = "PASS" if rep["grader_check"]["pass"] else "FAIL: re-grade the saved weights with regrade.py --engine pilot before reading"
    rep["verdicts"] = V
    Path(a.json).write_text(json.dumps(rep, indent=1) + "\n")

    if a.md:
        L = [f"# g1 seeds3 report ({rep['n_done']}/{rep['n_runs']} runs DONE, {rep['gpu_hours']} GPU-hours of wall time)", ""]
        L += ["| run | GSM8K | MATH-500 | L4-5 | stopped % | boxed % | mean tokens | L1 bpb | L3 bpb | HellaSwag |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r, e in table.items():
            L.append(f"| {r.replace('qwen2.5-0.5b__', '')} | {e['gsm8k']} | {e['math500']} | {e['math500_L45']} | {e.get('gsm8k_stopped_rate')} | "
                     f"{e.get('gsm8k_boxed_rate')} | {e.get('gsm8k_mean_tokens')} | {e['L1_heldout_bpb']} | {e['L3_heldout_bpb']} | {e['hellaswag']} |")
        L += ["", "## Readout (points; per seed, mean, 95% CI over items; t over seeds)", ""]
        for k, c in R.items():
            L.append(f"- {k}: {c.get('per_seed', '')} mean {c['mean']} CI {c['ci95_items']}" + (f" t {c['ci95_seeds_t']}" if "ci95_seeds_t" in c else ""))
        L += ["", "## Verdicts", ""] + [f"- {k}: {v}" for k, v in V.items()]
        if "grader_check" in rep:
            L += ["", f"Grader check: {json.dumps(rep['grader_check'])}"]
        Path(a.md).write_text("\n".join(L) + "\n")
    print(json.dumps({"n_done": rep["n_done"], "verdicts": V}, indent=1))


if __name__ == "__main__":
    main()
