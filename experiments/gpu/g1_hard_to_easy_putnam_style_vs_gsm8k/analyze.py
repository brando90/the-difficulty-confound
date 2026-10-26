"""g1 analysis: hard-to-easy gates, figure and results.json (PLAN.md Section 3 defines every gate).

    python analyze.py --out-root $OUT --tier main [--report-dir results_main]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from results import RunSet, acc_matrix, boot_ci, bpb_parts, fmt, verdict_dir, write_outputs  # noqa: E402

ARMS = ["E", "H", "M", "N", "Hrep"]
COLORS = {"E": "#0072B2", "H": "#D55E00", "M": "#009E73", "N": "#7F7F7F", "Hrep": "#CC79A7", "base": "#000000"}


def diff_stat(a: np.ndarray, b: np.ndarray):
    """seed-mean accuracy difference a - b in points, as a function of an item resample."""
    da, db = a.mean(0), b.mean(0)
    return lambda idx: 100 * (da[idx].mean() - db[idx].mean())


def gate_diff(a: np.ndarray, b: np.ndarray) -> dict:
    est = 100 * (a.mean() - b.mean())
    lo, hi = boot_ci(diff_stat(a, b), a.shape[1])
    seeds = [100 * (a[i].mean() - b[i].mean()) for i in range(a.shape[0])]
    return {"estimate_pts": est, "ci95": [lo, hi], "per_seed_pts": seeds, "n_items": int(a.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tier", default="main")
    ap.add_argument("--report-dir")
    a = ap.parse_args()
    rs = RunSet(a.out_root, a.tier, ["g1"])
    if rs.missing:
        print(f"WARNING: {len(rs.missing)} runs not DONE; analysing what exists is for monitoring only, never for the gates:")
        for m in rs.missing:
            print("  ", m)
    F = rs.final_step()
    tps = rs.tokens_per_step()
    arm = {k: rs.arm_runs("g1", k) for k in ARMS}
    have = {k: v for k, v in arm.items() if all(r in rs.rows for r in v.values())}

    levels = np.array(rs.row(next(iter(have["E"].values())), F, "math500", "acc")["levels"], dtype=float)
    hard = levels >= 4  # MATH-500 levels 4-5

    gsm = {k: acc_matrix(rs, have[k], F, "gsm8k") for k in have}
    m45 = {k: acc_matrix(rs, have[k], F, "math500", hard) for k in have}
    res: dict = {"plan": "g1", "tier": a.tier, "final_step": F, "final_tokens": F * tps, "missing_runs": rs.missing, "gates": {}, "table": {}}

    g = res["gates"]
    g["A1_easy_not_subsumed"] = gate_diff(gsm["E"], gsm["H"])
    x = g["A1_easy_not_subsumed"]
    x["verdict"] = verdict_dir(x["estimate_pts"], *x["ci95"], x["per_seed_pts"], thr_support=3.0, thr_falsify=1.0)
    x["rule"] = "GSM8K accuracy E - H at the final checkpoint. SUPPORTED if >= 3 points, interval lower bound > 0 and all seeds > 0; FALSIFIED (hard training subsumes the easy end) if the interval upper bound < 1 point"

    g["A2_hard_end_needs_hard_data"] = gate_diff(m45["H"], m45["E"])
    x = g["A2_hard_end_needs_hard_data"]
    x["verdict"] = verdict_dir(x["estimate_pts"], *x["ci95"], x["per_seed_pts"], thr_support=0.0, thr_falsify=0.0)
    x["rule"] = "MATH-500 levels 4-5 accuracy H - E. SUPPORTED if interval lower bound > 0 and all seeds > 0; FALSIFIED if the upper bound < 0"

    v1, v2 = g["A1_easy_not_subsumed"]["verdict"], g["A2_hard_end_needs_hard_data"]["verdict"]
    if v1 == "SUPPORTED" and v2 == "SUPPORTED":
        cross = "NOT A LADDER: each end is best served by its own data (crossover)"
    elif v1 == "FALSIFIED" and v2 == "SUPPORTED":
        cross = "LADDER: hard-only training matches easy-only on the easy end and wins on the hard end"
    elif v1 == "FALSIFIED" and v2 != "SUPPORTED":
        cross = "NO DIFFICULTY EFFECT: the two pools are interchangeable at this budget"
    else:
        cross = "UNRESOLVED"
    g["A_crossover"] = {"verdict": cross, "rule": "combines A1 and A2"}

    # A3: overtraining. (a) does more hard-only training lose GSM8K accuracy? (b) does repeating a small hard slice lose more?
    gsteps = rs.gen_steps()
    earlier = [s for s in gsteps if s < F]
    if earlier:
        hf = acc_matrix(rs, have["H"], F, "gsm8k")
        best_prev = max(earlier, key=lambda s: acc_matrix(rs, have["H"], s, "gsm8k").mean())
        hp = acc_matrix(rs, have["H"], best_prev, "gsm8k")
        g["A3a_more_hard_training_forgets"] = gate_diff(hp, hf)
        g["A3a_more_hard_training_forgets"]["compared_step"] = best_prev
        x = g["A3a_more_hard_training_forgets"]
        x["verdict"] = verdict_dir(x["estimate_pts"], *x["ci95"], x["per_seed_pts"], thr_support=2.0, thr_falsify=0.0)
        x["rule"] = "GSM8K accuracy of H at its best earlier generation checkpoint minus at the final one. SUPPORTED (forgetting) if >= 2 points, lower bound > 0, all seeds > 0; FALSIFIED if upper bound < 0 (still improving). The best earlier checkpoint is chosen on the seed-mean, which biases this upward; read with that in mind"
    if "Hrep" in have:
        g["A3b_repeated_hard_slice_forgets"] = gate_diff(gsm["H"], gsm["Hrep"])
        x = g["A3b_repeated_hard_slice_forgets"]
        x["verdict"] = verdict_dir(x["estimate_pts"], *x["ci95"], x["per_seed_pts"], thr_support=3.0, thr_falsify=0.0)
        x["rule"] = "GSM8K accuracy H - Hrep. SUPPORTED if >= 3 points, lower bound > 0, all seeds > 0; FALSIFIED if upper bound < 0"

    # descriptive contrasts (no verdicts)
    d = res["descriptive"] = {}
    for name, (p, q, mat) in {"M_minus_E_gsm8k": ("M", "E", gsm), "M_minus_H_math45": ("M", "H", m45),
                              "E_minus_N_gsm8k": ("E", "N", gsm), "H_minus_N_gsm8k": ("H", "N", gsm),
                              "E_minus_N_math45": ("E", "N", m45), "H_minus_N_math45": ("H", "N", m45)}.items():
        if p in have and q in have:
            d[name] = gate_diff(mat[p], mat[q])

    # table: final accuracies and bits per byte
    base_g = np.array(rs.base_row("gsm8k", "acc")["correct"], float)
    base_m = np.array(rs.base_row("math500", "acc")["correct"], float)[hard]
    res["table"]["base"] = {"gsm8k": 100 * base_g.mean(), "math500_L45": 100 * base_m.mean()}
    for k in have:
        row = {"gsm8k": 100 * gsm[k].mean(), "math500_L45": 100 * m45[k].mean(),
               "math500_all": 100 * acc_matrix(rs, have[k], F, "math500").mean()}
        for s in ("L1_heldout", "L3_heldout", "web_heldout"):
            bits = [bpb_parts(rs, rid, F, s) for rid in have[k].values()]
            row[f"bpb_{s}"] = float(np.mean([b.sum() / n.sum() for b, n in bits]))
        res["table"][k] = row

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k in have:
        xs = [s * tps / 1e6 for s in gsteps]
        ax[0].plot(xs, [100 * acc_matrix(rs, have[k], s, "gsm8k").mean() for s in gsteps], "o-", color=COLORS[k], label=k)
        ax[1].plot(xs, [100 * acc_matrix(rs, have[k], s, "math500", hard).mean() for s in gsteps], "o-", color=COLORS[k], label=k)
        cs = rs.ckpt_steps()
        for s_, ls in (("L1_heldout", "-"), ("L3_heldout", "--")):
            ys = []
            for st in cs:
                parts = [bpb_parts(rs, rid, st, s_) for rid in have[k].values()]
                ys.append(np.mean([b.sum() / n.sum() for b, n in parts]))
            ax[2].plot([st * tps / 1e6 for st in cs], ys, ls, color=COLORS[k], label=f"{k} on {s_[:2]}")
    ax[0].axhline(res["table"]["base"]["gsm8k"], color="black", ls=":", label="base")
    ax[1].axhline(res["table"]["base"]["math500_L45"], color="black", ls=":", label="base")
    ax[0].set(title="GSM8K accuracy (easy end)", xlabel="training tokens (M)", ylabel="% correct", xscale="log")
    ax[1].set(title="MATH-500 levels 4-5 accuracy (hard end)", xlabel="training tokens (M)", ylabel="% correct", xscale="log")
    ax[2].set(title="held-out bits/byte: L1 solid, L3 dashed", xlabel="training tokens (M)", ylabel="bits per byte", xscale="log")
    ax[0].legend(fontsize=8)
    ax[2].legend(fontsize=7, ncol=2)
    fig.suptitle(f"g1 hard-to-easy ({a.tier} tier): {cross}", fontsize=10)
    fig.tight_layout()
    rd = Path(a.report_dir) if a.report_dir else HERE / f"results_{a.tier}"
    rd.mkdir(parents=True, exist_ok=True)
    fig.savefig(rd / "g1_hard_to_easy.png", dpi=150)
    fig.savefig(rd / "g1_hard_to_easy.pdf")

    md = [f"# g1 results draft ({a.tier} tier)", "", f"Crossover: **{cross}**", "",
          "| Gate | Estimate (points) | 95% item interval | Per seed | Verdict |", "|---|---|---|---|---|"]
    for name, x in g.items():
        if "estimate_pts" in x:
            md.append(f"| {name} | {fmt(x['estimate_pts'])} | [{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}] | "
                      f"{', '.join(fmt(s) for s in x['per_seed_pts'])} | **{x['verdict']}** |")
    md += ["", "| Arm | GSM8K % | MATH-500 L4-5 % | MATH-500 all % | bpb L1 | bpb L3 | bpb web |", "|---|---|---|---|---|---|---|",
           f"| base | {fmt(res['table']['base']['gsm8k'], 1)} | {fmt(res['table']['base']['math500_L45'], 1)} | | | | |"]
    for k in have:
        t = res["table"][k]
        md.append(f"| {k} | {fmt(t['gsm8k'], 1)} | {fmt(t['math500_L45'], 1)} | {fmt(t['math500_all'], 1)} | "
                  f"{fmt(t['bpb_L1_heldout'], 3)} | {fmt(t['bpb_L3_heldout'], 3)} | {fmt(t['bpb_web_heldout'], 3)} |")
    write_outputs(rd, res, md)


if __name__ == "__main__":
    main()
