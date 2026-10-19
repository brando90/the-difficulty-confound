"""One shared run registry for the three GPU plans (g1, g2, g3).

A run is (model, mixture, budget, learning rate, seed). Its id depends only on
those, so g1's hard-only arm, g2's hard-only arm and g3's (L3, alpha = 1) cell
are the same run and are trained once. Each model also has one "base" run:
the untrained model evaluated at step 0.

    python runs.py list   --tier main --plans g1,g2,g3 [--secondary]
    python runs.py show   --tier main --plans g1 --index 3
    python runs.py status --tier main --plans g1,g2,g3 --out-root $OUT
    python runs.py cost   --tier main --plans g1,g2,g3 [--secondary]
    python runs.py select-lr --out-root $OUT [--write]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

GPU_DIR = Path(__file__).resolve().parent.parent
COMMON = GPU_DIR / "common"
PLAN_DIRS = {
    "g1": GPU_DIR / "g1_hard_to_easy_putnam_style_vs_gsm8k",
    "g2": GPU_DIR / "g2_cross_domain_math_vs_poetry",
    "g3": GPU_DIR / "g3_llm_complexity_vs_alignment_elasticity",
}
ALL_PLANS = ["g1", "g2", "g3"]


def load_suite() -> dict:
    return json.loads((COMMON / "suite.json").read_text())


def load_plan(name: str) -> dict:
    return json.loads((PLAN_DIRS[name] / "config.json").read_text())


def parse_plans(text: str | None) -> list[str]:
    plans = ALL_PLANS if not text or text == "all" else [p.strip() for p in text.split(",") if p.strip()]
    for p in plans:
        if p not in PLAN_DIRS:
            raise SystemExit(f"unknown plan {p!r}; choose from {ALL_PLANS}")
    return plans


def parse_seeds(text: str | None) -> list[int] | None:
    return None if not text or text == "all" else [int(x) for x in text.split(",")]


def mix_tag(mixture: dict, caps: dict | None) -> str:
    parts = [f"{pool}{frac:g}" for pool, frac in sorted(mixture.items()) if frac > 0]
    tag = "+".join(parts)
    for pool, cap in sorted((caps or {}).items()):
        tag += f"[cap-{pool}-{cap / 1e6:g}M]"
    return tag


def model_lr(suite: dict, model: str) -> float | None:
    lr = suite["selected_lr"].get(model)
    if lr is not None:
        return float(lr)
    base = suite["selected_lr"].get("qwen2.5-0.5b")
    if model != "qwen2.5-0.5b" and base is not None:
        return float(base) * 0.5  # suite["lr_scale_rule"]
    return None


def expand(tier: str, plans: list[str], secondary: bool = False, allow_unset_lr: bool = False,
           seeds: list[int] | None = None) -> list[dict]:
    """Every run the given plans need in this tier, merged by run id, sorted by id.
    `seeds` restricts the tier's seeds (a seed-0 first look reuses exactly the ids of the full tier)."""
    suite = load_suite()
    if tier not in suite["tiers"]:
        raise SystemExit(f"unknown tier {tier!r}; choose from {list(suite['tiers'])}")
    t = suite["tiers"][tier]
    runs: dict[str, dict] = {}

    def add(model: str, arm_name: str, arm: dict, plan: str, lr: float, seed: int) -> None:
        tag = mix_tag(arm["mixture"], arm.get("caps"))
        budget = f"T{t['max_steps']}x{t['seqs_per_step']}x{t['seq_len']}"
        rid = f"{model}__{tag}__{budget}__lr{lr:g}__s{seed}"
        spec = runs.get(rid)
        if spec is None:
            spec = runs[rid] = {
                "run_id": rid, "kind": "train", "tier": tier, "model": model,
                "hf": suite["models"][model]["hf"], "micro_batch": suite["models"][model]["micro_batch"],
                "mixture": {k: float(v) for k, v in arm["mixture"].items() if v > 0},
                "caps": {k: int(v) for k, v in (arm.get("caps") or {}).items()},
                "lr": lr, "seed": seed, "seq_len": t["seq_len"], "seqs_per_step": t["seqs_per_step"],
                "max_steps": t["max_steps"], "ckpt_steps": t["ckpt_steps"], "gen_steps": [],
                "gen_max_new_tokens": t.get("gen_max_new_tokens", 768), "eval_limit": t.get("eval_limit"),
                "device": t.get("device", "cuda"), "optim": suite["optim"], "format": suite["format"],
                "used_by": [],
            }
        spec["used_by"].append(f"{plan}:{arm_name}")
        if arm.get("gen"):
            spec["gen_steps"] = sorted(set(spec["gen_steps"]) | set(t["gen_steps"]))

    for model in t["models"]:
        base_id = f"{model}__base" + ("_smoke" if tier == "smoke" else "")
        runs[base_id] = {
            "run_id": base_id, "kind": "base", "tier": "any", "model": model, "hf": suite["models"][model]["hf"],
            "micro_batch": suite["models"][model]["micro_batch"], "seed": 0, "seq_len": t["seq_len"],
            "ckpt_steps": [0], "gen_steps": [0], "gen_max_new_tokens": t.get("gen_max_new_tokens", 768),
            "eval_limit": t.get("eval_limit"), "device": t.get("device", "cuda"), "format": suite["format"],
            "used_by": [f"{p}:base" for p in plans],
        }
        for plan in plans:
            cfg = load_plan(plan)
            pt = cfg["tiers"].get(tier, {})
            names = list(pt.get("arms", [])) + (list(pt.get("secondary", [])) if secondary else [])
            for seed in [x for x in t["seeds"] if seeds is None or x in seeds]:
                for arm_name in pt.get("sweep", []):
                    for lr in t["lr_sweep"]:
                        add(model, arm_name, cfg["arms"][arm_name], plan, float(lr), seed)
                for arm_name in names:
                    if "lr" in t:
                        lr = float(t["lr"])
                    elif "lr_default" in t:
                        lr = float(t["lr_default"])
                    else:
                        lr = model_lr(suite, model)
                        if lr is None:
                            if not allow_unset_lr:
                                raise SystemExit(
                                    f"selected_lr for {model} is not set in common/suite.json: run the pilot, then "
                                    "'runs.py select-lr --write' and commit it (the pilot disclosure commit)")
                            lr = float("nan")
                    add(model, arm_name, cfg["arms"][arm_name], plan, lr, seed)
    return [runs[k] for k in sorted(runs)]


def run_dir(out_root: str | Path, rid: str) -> Path:
    return Path(out_root) / "runs" / rid


def is_done(out_root: str | Path, rid: str) -> bool:
    return (run_dir(out_root, rid) / "DONE").exists()


def a100_hours(spec: dict, suite: dict, mfu: float = 0.35) -> dict:
    """Rough A100-80GB cost. Training: 6 * params * tokens FLOPs at `mfu` of 312 TFLOP/s bf16.
    Generation: GSM8K + MATH-500 (1,819 prompts) per generation checkpoint, measured rule of thumb
    of 0.10 h per 0.5B model per checkpoint with HF generate, scaled linearly in parameters.
    Bits-per-byte evaluations: 7 sets of about 1,000 items per checkpoint, forward only (small)."""
    p = suite["models"][spec["model"]]["params"]
    if spec["kind"] == "base":
        tokens = 0
    else:
        tokens = spec["max_steps"] * spec["seqs_per_step"] * spec["seq_len"]
    train_h = 6 * p * tokens / (312e12 * mfu) / 3600
    gen_h = len(spec["gen_steps"]) * 0.10 * (p / 0.494e9)
    bpb_h = len(spec["ckpt_steps"]) * 0.004 * (p / 0.494e9)
    return {"train": train_h, "gen": gen_h, "bpb": bpb_h, "total": train_h + gen_h + bpb_h + 0.05}


def cmd_list(a) -> None:
    for i, s in enumerate(expand(a.tier, parse_plans(a.plans), a.secondary, allow_unset_lr=True, seeds=parse_seeds(a.seeds))):
        print(f"{i}\t{s['run_id']}\t{','.join(s['used_by'])}")


def cmd_show(a) -> None:
    runs = expand(a.tier, parse_plans(a.plans), a.secondary, seeds=parse_seeds(a.seeds))
    if not 0 <= a.index < len(runs):
        raise SystemExit(f"index {a.index} outside 0..{len(runs) - 1}")
    print(json.dumps(runs[a.index], indent=2))


def cmd_count(a) -> None:
    print(len(expand(a.tier, parse_plans(a.plans), a.secondary, allow_unset_lr=True, seeds=parse_seeds(a.seeds))))


def cmd_status(a) -> None:
    runs = expand(a.tier, parse_plans(a.plans), a.secondary, allow_unset_lr=True, seeds=parse_seeds(a.seeds))
    done = [s for s in runs if is_done(a.out_root, s["run_id"])]
    claimed = [s for s in runs if not is_done(a.out_root, s["run_id"]) and (Path(a.out_root) / "claims" / s["run_id"]).exists()]
    print(f"{len(done)}/{len(runs)} DONE, {len(claimed)} claimed but not DONE (running, or crashed)")
    for s in claimed:
        print("  claimed:", s["run_id"])


def cmd_cost(a) -> None:
    suite = load_suite()
    runs = expand(a.tier, parse_plans(a.plans), a.secondary, allow_unset_lr=True, seeds=parse_seeds(a.seeds))
    tot = {"train": 0.0, "gen": 0.0, "bpb": 0.0, "total": 0.0}
    for s in runs:
        c = a100_hours(s, suite, a.mfu)
        for k in tot:
            tot[k] += c[k]
    print(f"{len(runs)} runs; A100-80GB hours at MFU {a.mfu}: train {tot['train']:.1f}, generation {tot['gen']:.1f}, "
          f"bits-per-byte {tot['bpb']:.1f}, total with overhead {tot['total']:.1f}")


def cmd_select_lr(a) -> None:
    """Pilot rule (pre-registered): for each swept learning rate, average the held-out bits per byte of the
    E run on L1_heldout and of the H run on L3_heldout at the last pilot checkpoint; pick the lowest. Ties
    within 0.002 bits/byte go to the smaller rate. Any non-finite loss disqualifies a rate."""
    suite = load_suite()
    t = suite["tiers"]["pilot"]
    scores = {}
    for lr in t["lr_sweep"]:
        vals = []
        for pool, ev in (("L1_gradeschool", "L1_heldout"), ("L3_olympiad", "L3_heldout")):
            rid = f"qwen2.5-0.5b__{pool}1__T{t['max_steps']}x{t['seqs_per_step']}x{t['seq_len']}__lr{float(lr):g}__s0"
            m = run_dir(a.out_root, rid) / "metrics.jsonl"
            if not is_done(a.out_root, rid):
                raise SystemExit(f"pilot run {rid} is not DONE")
            rows = [json.loads(x) for x in m.read_text().splitlines()]
            last = max(r["step"] for r in rows)
            v = [r["bpb"] for r in rows if r["step"] == last and r["set"] == ev and r["kind"] == "bpb"][0]
            vals.append(v)
        score = sum(vals) / 2
        scores[float(lr)] = score if all(math.isfinite(v) for v in vals) else float("inf")
        print(f"lr {float(lr):g}: E on L1 {vals[0]:.4f}, H on L3 {vals[1]:.4f}, mean {score:.4f} bits/byte")
    best = min(scores.values())
    chosen = min(lr for lr, s in scores.items() if s <= best + 0.002)
    print(f"selected learning rate for qwen2.5-0.5b: {chosen:g} (1.5B rule: {chosen * 0.5:g})")
    if a.write:
        suite["selected_lr"]["qwen2.5-0.5b"] = chosen
        (COMMON / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
        print("written to common/suite.json; commit it as the pilot disclosure commit, with this output in CKPT.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("list", cmd_list), ("show", cmd_show), ("count", cmd_count), ("status", cmd_status), ("cost", cmd_cost)):
        p = sub.add_parser(name)
        p.add_argument("--tier", required=True)
        p.add_argument("--plans", default="all")
        p.add_argument("--secondary", action="store_true", help="include g3's within-family distractor arms")
        p.add_argument("--seeds", default=os.environ.get("SEEDS", "all"), help="e.g. 0 for a seed-0 first look")
        if name == "show":
            p.add_argument("--index", type=int, required=True)
        if name == "status":
            p.add_argument("--out-root", default=os.environ.get("OUT", "runs_out"))
        if name == "cost":
            p.add_argument("--mfu", type=float, default=0.35)
        p.set_defaults(fn=fn)
    p = sub.add_parser("select-lr")
    p.add_argument("--out-root", default=os.environ.get("OUT", "runs_out"))
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_select_lr)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
