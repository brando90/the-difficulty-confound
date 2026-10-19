"""g3 analysis: T*(K, alpha), elasticities, the ratio E = e_alpha / e_K, figure and results.json.

    python analyze.py --out-root $OUT --tier main [--secondary] [--report-dir results_main]

For target level k (held-out set Lk_heldout), arm (k, alpha) and seed s, the improvement after T training tokens is
    D(T) = bpb_base(k) - bpb(k, alpha, s, T).
The level's reference improvement is D_max(k) = the median over seeds of D at (k, alpha = 1, final checkpoint).
T*(k, alpha, s) is the first T with D(T) >= q * D_max(k), q = 0.5, interpolated linearly in log T between
checkpoints. It is left-censored if the first checkpoint already passes and right-censored if the last does not.
Elasticities come from ordinary least squares on uncensored cells:
    log T* = c + e_K log K - e_alpha log alpha,     E = e_alpha / e_K.
K is the level's complexity proxy from the data manifest (median held-out solution length in tokens).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from results import RunSet, bpb_parts, fmt, write_outputs  # noqa: E402

Q_PRIMARY = 0.5
Q_SENS = [0.25, 0.5, 0.75]
SYNTHETIC_E = 0.80  # Exp 13-small, within-family GINC HMMs


def t_star(tokens: list[float], d: list[float], thr: float) -> tuple[float, str]:
    if d[0] >= thr:
        return tokens[0], "left"
    for i in range(1, len(tokens)):
        if d[i] >= thr:
            x0, x1 = math.log(tokens[i - 1]), math.log(tokens[i])
            f = (thr - d[i - 1]) / (d[i] - d[i - 1])
            return math.exp(x0 + f * (x1 - x0)), "ok"
    return tokens[-1], "right"


def ols(rows: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """rows of (log K, log alpha, log T*) -> (c, e_K, e_alpha) with log T* = c + e_K logK - e_alpha log alpha."""
    if len(rows) < 4 or len({r[0] for r in rows}) < 2 or len({r[1] for r in rows}) < 2:
        return None
    X = np.array([[1.0, r[0], -r[1]] for r in rows])
    y = np.array([r[2] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1]), float(beta[2])


def analyse(rs: RunSet, cfg: dict, K: dict, distractor: str, q: float) -> dict:
    F = rs.final_step()
    tps = rs.tokens_per_step()
    cs = rs.ckpt_steps()
    toks = [s * tps for s in cs]
    arms = {name: arm for name, arm in cfg["arms"].items()
            if arm["distractor"] == distractor or (distractor != "web" and arm["alpha"] == 1.0 and arm["distractor"] == "web")}
    base = {lvl: bpb_parts(rs, rs.base["run_id"], 0, f"{lvl}_heldout") for lvl in cfg["levels"]}
    base_bpb = {lvl: base[lvl][0].sum() / base[lvl][1].sum() for lvl in cfg["levels"]}

    curves = {}
    for name, arm in arms.items():
        for seed, rid in rs.arm_runs("g3", name).items():
            if rid not in rs.rows:
                continue
            lvl = arm["level"]
            d = []
            for s in cs:
                b, n = bpb_parts(rs, rid, s, f"{lvl}_heldout")
                d.append(base_bpb[lvl] - b.sum() / n.sum())
            curves[(lvl, arm["alpha"], seed)] = d
    dmax = {}
    for lvl in cfg["levels"]:
        finals = [c[-1] for (l_, a_, _), c in curves.items() if l_ == lvl and a_ == 1.0]
        dmax[lvl] = float(np.median(finals)) if finals else float("nan")
    cells = []
    for (lvl, alpha, seed), d in sorted(curves.items()):
        thr = q * dmax[lvl]
        ts, cens = t_star(toks, d, thr) if math.isfinite(thr) and thr > 0 else (float("nan"), "no_reference")
        cells.append({"level": lvl, "alpha": alpha, "seed": seed, "K": K[lvl], "T_star": ts, "censored": cens, "D_final": d[-1], "threshold": thr})
    seeds = sorted({c["seed"] for c in cells})

    def fit(sel):
        rows = [(math.log(c["K"]), math.log(c["alpha"]), math.log(c["T_star"])) for c in sel if c["censored"] == "ok"]
        b = ols(rows)
        if b is None:
            return {"n_cells": len(rows), "e_K": float("nan"), "e_alpha": float("nan"), "E": float("nan")}
        return {"n_cells": len(rows), "c": b[0], "e_K": b[1], "e_alpha": b[2], "E": b[2] / b[1] if b[1] != 0 else float("nan")}

    pooled_rows = {}
    for c in cells:
        pooled_rows.setdefault((c["level"], c["alpha"]), []).append(c)
    pooled = []
    for (lvl, alpha), cs_ in pooled_rows.items():
        ok = [c for c in cs_ if c["censored"] == "ok"]
        if len(ok) == len(cs_):
            pooled.append({"level": lvl, "alpha": alpha, "K": K[lvl], "censored": "ok",
                           "T_star": math.exp(np.mean([math.log(c["T_star"]) for c in ok]))})
        else:
            pooled.append({"level": lvl, "alpha": alpha, "K": K[lvl], "censored": "some_seed_censored", "T_star": float("nan")})
    out = {"q": q, "distractor": distractor, "D_max": dmax, "base_bpb": base_bpb, "cells": cells, "pooled_cells": pooled,
           "pooled_fit": fit(pooled), "per_seed_fit": {s: fit([c for c in cells if c["seed"] == s]) for s in seeds},
           "censored_fraction": sum(c["censored"] != "ok" for c in cells) / max(len(cells), 1)}
    # C1: complexity multiplier at alpha = 1, hardest over easiest level
    lv = cfg["levels"]
    mk = {}
    for s in seeds:
        a1 = {c["level"]: c for c in cells if c["seed"] == s and c["alpha"] == 1.0}
        if lv[0] in a1 and lv[-1] in a1:
            mk[s] = {"M_K": a1[lv[-1]]["T_star"] / a1[lv[0]]["T_star"], "censoring": [a1[lv[0]]["censored"], a1[lv[-1]]["censored"]]}
    out["M_K_per_seed"] = mk
    return out


def verdicts(r: dict) -> dict:
    v = {}
    mks = [x["M_K"] for x in r["M_K_per_seed"].values()]
    # left-censored easy level makes M_K a lower bound; right-censored hard level also a lower bound
    if mks and all(m >= 2 for m in mks):
        v["C1_complexity_costs_data"] = "SUPPORTED"
    elif mks and all(m <= 1 for m in mks):
        v["C1_complexity_costs_data"] = "FALSIFIED"
    else:
        v["C1_complexity_costs_data"] = "UNRESOLVED"
    E = r["pooled_fit"]["E"]
    Es = [f["E"] for f in r["per_seed_fit"].values()]
    if v["C1_complexity_costs_data"] != "SUPPORTED":  # PLAN.md: without C1, e_K is not identified and E means nothing
        v["C2_alignment_dominates"] = "UNRESOLVED (C1 not supported, so e_K is not identified)"
    elif r["censored_fraction"] > 1 / 3:
        v["C2_alignment_dominates"] = f"UNRESOLVED (censored fraction {r['censored_fraction']:.2f} > 1/3)"
    elif math.isfinite(E) and E > 1 and Es and all(math.isfinite(x) and x > 1 for x in Es):
        v["C2_alignment_dominates"] = "SUPPORTED"
    elif math.isfinite(E) and E < 1 and Es and all(math.isfinite(x) and x < 1 for x in Es):
        v["C2_alignment_dominates"] = "REVERSED (complexity dominates, as in Exp 13-small)"
    else:
        v["C2_alignment_dominates"] = "UNRESOLVED"
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data-root", required=True, help="for the complexity proxy K in the data manifest")
    ap.add_argument("--tier", default="main")
    ap.add_argument("--secondary", action="store_true", help="also analyse the within-family distractor arms")
    ap.add_argument("--report-dir")
    a = ap.parse_args()
    cfg = json.loads((HERE / "config.json").read_text())
    sys.path.insert(0, str(HERE.parent / "common"))
    import runs

    manifest = json.loads((Path(a.data_root) / runs.load_suite()["data_version"] / "manifest.json").read_text())
    K = {lvl: float(v["median_tokens"]) for lvl, v in manifest["complexity_K"]["levels"].items()}
    rs = RunSet(a.out_root, a.tier, ["g3"], a.secondary)
    if rs.missing:
        print(f"WARNING: {len(rs.missing)} runs not DONE; monitoring only, never for the gates")
    res = {"plan": "g3", "tier": a.tier, "K": K, "K_proxy": manifest["complexity_K"]["proxy"], "missing_runs": rs.missing, "synthetic_E": SYNTHETIC_E}
    prim = analyse(rs, cfg, K, "web", Q_PRIMARY)
    res["primary"] = prim
    res["verdicts"] = verdicts(prim)
    res["sensitivity_q"] = {str(q): analyse(rs, cfg, K, "web", q)["pooled_fit"] for q in Q_SENS}
    # ordinal K robustness: K = 1, 2, 3
    res["sensitivity_ordinal_K"] = analyse(rs, cfg, {lvl: float(i + 1) for i, lvl in enumerate(cfg["levels"])}, "web", Q_PRIMARY)["pooled_fit"]
    if a.secondary:
        w = analyse(rs, cfg, K, "other_levels", Q_PRIMARY)
        res["within_family"] = w
        res["within_family_verdicts"] = verdicts(w)
        res["C3_within_vs_web"] = {"E_web": prim["pooled_fit"]["E"], "E_within": w["pooled_fit"]["E"],
                                   "prediction": "E_within < E_web (within-family distractors share structure with the target)"}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tps = rs.tokens_per_step()
    toks = [s * tps / 1e6 for s in rs.ckpt_steps()]
    colors = {"L1": "#0072B2", "L2": "#009E73", "L3": "#D55E00"}
    alphas = sorted({c["alpha"] for c in prim["cells"]})
    for lvl in cfg["levels"]:
        ys = [next((p["T_star"] for p in prim["pooled_cells"] if p["level"] == lvl and p["alpha"] == al), float("nan")) for al in alphas]
        ax[0].plot(alphas, np.array(ys) / 1e6, "o-", color=colors[lvl], label=f"{lvl} (K={K[lvl]:.0f})")
    ax[0].set(xscale="log", yscale="log", xlabel="alignment alpha (target share)", ylabel="T* (M tokens)", title=f"tokens to reach {Q_PRIMARY:g} of the alpha=1 gain")
    ax[0].legend(fontsize=8)
    # improvement curves (seed mean) at alpha = 1 and the smallest alpha
    for lvl in cfg["levels"]:
        for al, ls in ((1.0, "-"), (alphas[0], ":")):
            rids = [rid for name, arm in cfg["arms"].items() if arm["level"] == lvl and arm["alpha"] == al and arm["distractor"] == "web"
                    for rid in rs.arm_runs("g3", name).values() if rid in rs.rows]
            if not rids:
                continue
            ys = []
            for s in rs.ckpt_steps():
                vals = []
                for rid in rids:
                    b, n = bpb_parts(rs, rid, s, f"{lvl}_heldout")
                    vals.append(prim["base_bpb"][lvl] - b.sum() / n.sum())
                ys.append(np.mean(vals))
            ax[1].plot(toks, ys, ls, color=colors[lvl], label=f"{lvl} alpha={al:g}")
    ax[1].set(xscale="log", xlabel="training tokens (M)", ylabel="bits/byte gained over base", title="improvement on each level's held-out set")
    ax[1].legend(fontsize=7, ncol=2)
    Es = [f["E"] for f in prim["per_seed_fit"].values()]
    ax[2].scatter(range(len(Es)), Es, color="#555555", label="per seed")
    ax[2].axhline(prim["pooled_fit"]["E"], color="#D55E00", label=f"pooled E = {fmt(prim['pooled_fit']['E'])}")
    ax[2].axhline(1.0, color="black", ls="--", label="E = 1 (alignment dominates above)")
    ax[2].axhline(SYNTHETIC_E, color="#0072B2", ls=":", label="Exp 13-small synthetic 0.80")
    ax[2].set(title="elasticity ratio E = e_alpha / e_K", xlabel="seed", ylabel="E")
    ax[2].set_xticks(range(len(Es)))
    ax[2].legend(fontsize=7)
    fig.suptitle(f"g3 LLM complexity vs alignment ({a.tier} tier): C2 {res['verdicts']['C2_alignment_dominates']}", fontsize=10)
    fig.tight_layout()
    rd = Path(a.report_dir) if a.report_dir else HERE / f"results_{a.tier}"
    rd.mkdir(parents=True, exist_ok=True)
    fig.savefig(rd / "g3_elasticity.png", dpi=150)
    fig.savefig(rd / "g3_elasticity.pdf")

    pf = prim["pooled_fit"]
    md = [f"# g3 results draft ({a.tier} tier)", "",
          f"K proxy: {res['K_proxy']}: " + ", ".join(f"{k} = {v:g}" for k, v in K.items()), "",
          "| Quantity | Pooled | Per seed |", "|---|---|---|",
          f"| e_K | {fmt(pf['e_K'], 3)} | {', '.join(fmt(f['e_K'], 3) for f in prim['per_seed_fit'].values())} |",
          f"| e_alpha | {fmt(pf['e_alpha'], 3)} | {', '.join(fmt(f['e_alpha'], 3) for f in prim['per_seed_fit'].values())} |",
          f"| E = e_alpha / e_K | **{fmt(pf['E'], 3)}** | {', '.join(fmt(f['E'], 3) for f in prim['per_seed_fit'].values())} |",
          f"| M_K (L3 over L1 at alpha = 1) | | {', '.join(fmt(x['M_K'], 2) for x in prim['M_K_per_seed'].values())} |",
          f"| censored cells | {prim['censored_fraction']:.2f} | |", "",
          "| Gate | Verdict |", "|---|---|"]
    md += [f"| {k} | **{v}** |" for k, v in res["verdicts"].items()]
    md += ["", "Sensitivity of E: " + ", ".join(f"q={q}: {fmt(f['E'], 3)}" for q, f in res["sensitivity_q"].items())
           + f"; ordinal K: {fmt(res['sensitivity_ordinal_K']['E'], 3)}. Synthetic Exp 13-small: {SYNTHETIC_E}."]
    if a.secondary:
        md.append(f"Within-family distractors: E = {fmt(res['within_family']['pooled_fit']['E'], 3)} ({res['within_family_verdicts']['C2_alignment_dominates']}).")
    write_outputs(rd, res, md)


if __name__ == "__main__":
    main()
