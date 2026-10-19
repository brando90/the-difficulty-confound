"""g2 analysis: cross-domain transfer gates, figure and results.json (PLAN.md Section 3 defines every gate).

    python analyze.py --out-root $OUT --tier main [--report-dir results_main]

Transfer ratio on held-out poetry, for arm X at the final checkpoint:
    R_X = (bpb_base - bpb_X) / (bpb_base - bpb_P)
0 means X bought nothing for poetry, 1 means as much as training on poetry itself, below 0 means X made poetry worse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from results import RunSet, acc_matrix, boot_ci, bpb_parts, fmt, write_outputs  # noqa: E402

ARMS = ["E", "H", "N", "P"]
COLORS = {"E": "#0072B2", "H": "#D55E00", "N": "#7F7F7F", "P": "#009E73"}


def bits_matrix(rs: RunSet, seeds_to_rid: dict, step: int, set_: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [bpb_parts(rs, rid, step, set_) for rid in seeds_to_rid.values()]
    return np.array([p[0] for p in parts]), parts[0][1]  # (seeds x items) bits, bytes per item


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tier", default="main")
    ap.add_argument("--report-dir")
    a = ap.parse_args()
    rs = RunSet(a.out_root, a.tier, ["g2"])
    if rs.missing:
        print(f"WARNING: {len(rs.missing)} runs not DONE; monitoring only, never for the gates")
    F, tps = rs.final_step(), rs.tokens_per_step()
    arm = {k: rs.arm_runs("g2", k) for k in ARMS}
    have = {k: v for k, v in arm.items() if all(r in rs.rows for r in v.values())}
    base_bits, nbytes = bpb_parts(rs, rs.base["run_id"], 0, "poetry_heldout")
    bits = {k: bits_matrix(rs, have[k], F, "poetry_heldout")[0] for k in have}
    seeds = list(have["P"].keys())

    def ratio(x_bits, p_bits, idx=None):
        idx = slice(None) if idx is None else idx
        b0 = base_bits[idx].sum() / nbytes[idx].sum()
        bx = x_bits[..., idx].sum(-1) / nbytes[idx].sum()
        bp = p_bits[..., idx].sum(-1) / nbytes[idx].sum()
        return (b0 - bx) / (b0 - bp)

    res: dict = {"plan": "g2", "tier": a.tier, "final_step": F, "final_tokens": F * tps, "missing_runs": rs.missing,
                 "base_bpb_poetry": float(base_bits.sum() / nbytes.sum()), "gates": {}, "transfer_ratio": {}, "table": {}}
    n = len(nbytes)
    for k in ("E", "H", "N"):
        if k in have:
            per_seed = [float(ratio(bits[k][i], bits["P"][i])) for i in range(len(seeds))]
            est = float(ratio(bits[k].mean(0), bits["P"].mean(0)))
            lo, hi = boot_ci(lambda idx: ratio(bits[k].mean(0), bits["P"].mean(0), idx), n)
            res["transfer_ratio"][k] = {"estimate": est, "ci95": [lo, hi], "per_seed": per_seed}

    g = res["gates"]
    t = res["transfer_ratio"]["H"]
    if t["estimate"] <= 0.10 and t["ci95"][1] < 0.25:
        v = "SUPPORTED"
    elif t["ci95"][0] > 0.25:
        v = "FALSIFIED"
    else:
        v = "UNRESOLVED"
    g["B1_hard_maths_buys_no_poetry"] = {**t, "verdict": v, "rule": "R_H <= 0.10 with interval upper bound < 0.25: SUPPORTED; interval lower bound > 0.25: FALSIFIED"}

    if "E" in have:
        dseed = [h - e for h, e in zip(res["transfer_ratio"]["H"]["per_seed"], res["transfer_ratio"]["E"]["per_seed"])]
        est = res["transfer_ratio"]["H"]["estimate"] - res["transfer_ratio"]["E"]["estimate"]
        lo, hi = boot_ci(lambda idx: ratio(bits["H"].mean(0), bits["P"].mean(0), idx) - ratio(bits["E"].mean(0), bits["P"].mean(0), idx), n)
        if hi <= 0.05:
            v = "SUPPORTED"
        elif lo > 0.05 and all(x > 0.05 for x in dseed):
            v = "FALSIFIED"
        else:
            v = "UNRESOLVED"
        g["B2_harder_maths_no_more_general"] = {"estimate": est, "ci95": [lo, hi], "per_seed": dseed, "verdict": v,
                                                "rule": "R_H - R_E: interval upper bound <= 0.05: SUPPORTED (difficulty buys no generality); lower bound > 0.05 and all seeds > 0.05: FALSIFIED"}

    # B3 (descriptive prediction): maths-only training makes poetry worse than the web-text control does
    if "N" in have:
        res["descriptive_B3"] = {"R_H": res["transfer_ratio"]["H"]["estimate"], "R_E": res["transfer_ratio"]["E"]["estimate"],
                                 "R_N": res["transfer_ratio"]["N"]["estimate"],
                                 "prediction": "R_H < 0 and R_H < R_N (maths-only training costs poetry; web text does not)"}

    # B4: reverse direction and a general benchmark, descriptive with item intervals
    d = res["descriptive_B4"] = {}
    for set_ in ("gsm8k", "math500"):
        b = np.array(rs.base_row(set_, "acc")["correct"], float)
        pm = acc_matrix(rs, have["P"], F, set_)
        lo, hi = boot_ci(lambda idx: 100 * (pm.mean(0)[idx].mean() - b[idx].mean()), len(b))
        d[f"P_minus_base_{set_}_pts"] = {"estimate": 100 * (pm.mean() - b.mean()), "ci95": [lo, hi]}
    hs_base = np.array(rs.base_row("hellaswag", "mc")["correct"], float)
    for k in have:
        m = np.array([rs.row(rid, F, "hellaswag", "mc")["correct"] for rid in have[k].values()], float)
        lo, hi = boot_ci(lambda idx: 100 * (m.mean(0)[idx].mean() - hs_base[idx].mean()), len(hs_base))
        d[f"{k}_minus_base_hellaswag_pts"] = {"estimate": 100 * (m.mean() - hs_base.mean()), "ci95": [lo, hi]}

    for k in have:
        res["table"][k] = {"bpb_poetry": float((bits[k].sum(1) / nbytes.sum()).mean()),
                           "gsm8k": 100 * acc_matrix(rs, have[k], F, "gsm8k").mean(),
                           "hellaswag": 100 * float(np.mean([rs.row(r, F, "hellaswag", "mc")["acc_norm"] for r in have[k].values()]))}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cs = rs.ckpt_steps()
    for k in have:
        ys = [float(np.mean([bpb_parts(rs, r, s, "poetry_heldout")[0].sum() / nbytes.sum() for r in have[k].values()])) for s in cs]
        ax[0].plot([s * tps / 1e6 for s in cs], ys, "o-", color=COLORS[k], label=k)
    ax[0].axhline(res["base_bpb_poetry"], color="black", ls=":", label="base")
    ax[0].set(title="held-out poetry, bits per byte", xlabel="training tokens (M)", ylabel="bits per byte", xscale="log")
    ax[0].legend(fontsize=8)
    ks = [k for k in ("E", "H", "N") if k in res["transfer_ratio"]]
    est = [res["transfer_ratio"][k]["estimate"] for k in ks]
    err = np.array([[e - res["transfer_ratio"][k]["ci95"][0], res["transfer_ratio"][k]["ci95"][1] - e] for k, e in zip(ks, est)]).T
    ax[1].bar(ks, est, yerr=err, color=[COLORS[k] for k in ks], capsize=4)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].axhline(0.25, color="black", ls="--", lw=0.8, label="falsification line 0.25")
    ax[1].set(title="transfer ratio to poetry (P = 1)", ylabel="R")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"g2 cross-domain ({a.tier} tier): B1 {g['B1_hard_maths_buys_no_poetry']['verdict']}", fontsize=10)
    fig.tight_layout()
    rd = Path(a.report_dir) if a.report_dir else HERE / f"results_{a.tier}"
    rd.mkdir(parents=True, exist_ok=True)
    fig.savefig(rd / "g2_cross_domain.png", dpi=150)
    fig.savefig(rd / "g2_cross_domain.pdf")

    md = [f"# g2 results draft ({a.tier} tier)", "", f"Base model poetry bits/byte: {fmt(res['base_bpb_poetry'], 4)}", "",
          "| Gate | Estimate | 95% item interval | Per seed | Verdict |", "|---|---|---|---|---|"]
    for name, x in g.items():
        md.append(f"| {name} | {fmt(x['estimate'], 3)} | [{fmt(x['ci95'][0], 3)}, {fmt(x['ci95'][1], 3)}] | "
                  f"{', '.join(fmt(s, 3) for s in x['per_seed'])} | **{x['verdict']}** |")
    md += ["", "| Arm | R (poetry) | poetry bpb | GSM8K % | HellaSwag % |", "|---|---|---|---|---|"]
    for k in have:
        r = res["transfer_ratio"].get(k, {"estimate": 1.0})["estimate"]
        md.append(f"| {k} | {fmt(r, 3)} | {fmt(res['table'][k]['bpb_poetry'], 4)} | {fmt(res['table'][k]['gsm8k'], 1)} | {fmt(res['table'][k]['hellaswag'], 1)} |")
    write_outputs(rd, res, md)


if __name__ == "__main__":
    main()
