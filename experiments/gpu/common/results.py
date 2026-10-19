"""Load finished runs and the statistics the three analyses share.

Uncertainty: every gate reports (a) the seed-mean estimate, (b) a 95% paired bootstrap interval over
evaluation items (10,000 draws, fixed seed), which covers the finite test set, and (c) each seed's own
estimate, because three seeds are too few for a seed bootstrap. A gate that needs a direction requires
every seed to agree with it, which covers training noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import runs

BOOT_DRAWS = 10_000
BOOT_SEED = 20261018


class RunSet:
    def __init__(self, out_root: str | Path, tier: str, plans: list[str], secondary: bool = False):
        self.out_root, self.tier = Path(out_root), tier
        self.specs = runs.expand(tier, plans, secondary, allow_unset_lr=True)
        self.rows: dict[str, list[dict]] = {}
        self.missing = []
        for s in self.specs:
            d = runs.run_dir(out_root, s["run_id"])
            if (d / "DONE").exists():
                self.rows[s["run_id"]] = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
            else:
                self.missing.append(s["run_id"])
        self.base = next(s for s in self.specs if s["kind"] == "base")

    def arm_runs(self, plan: str, arm: str) -> dict[int, str]:
        """seed -> run id for one arm of one plan."""
        out = {}
        for s in self.specs:
            if f"{plan}:{arm}" in s["used_by"]:
                out[s["seed"]] = s["run_id"]
        return dict(sorted(out.items()))

    def row(self, rid: str, step: int, set_: str, kind: str) -> dict:
        for r in self.rows.get(rid, []):
            if r["step"] == step and r["set"] == set_ and r["kind"] == kind:
                return r
        raise KeyError(f"{rid}: no {kind} row for {set_} at step {step}")

    def base_row(self, set_: str, kind: str) -> dict:
        return self.row(self.base["run_id"], 0, set_, kind)

    def final_step(self) -> int:
        return next(s for s in self.specs if s["kind"] == "train")["max_steps"]

    def gen_steps(self) -> list[int]:
        return sorted({x for s in self.specs if s["kind"] == "train" for x in s["gen_steps"]})

    def ckpt_steps(self) -> list[int]:
        return next(s for s in self.specs if s["kind"] == "train")["ckpt_steps"]

    def tokens_per_step(self) -> int:
        s = next(s for s in self.specs if s["kind"] == "train")
        return s["seqs_per_step"] * s["seq_len"]


def boot_ci(stat, n_items: int, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile interval of stat(idx) over item resamples; stat takes an index array."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n_items, n_items)) for _ in range(draws)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def acc_matrix(rs: RunSet, seeds_to_rid: dict[int, str], step: int, set_: str, mask: np.ndarray | None = None) -> np.ndarray:
    """(seeds x items) 0/1 correctness."""
    m = np.array([rs.row(rid, step, set_, "acc")["correct"] for rid in seeds_to_rid.values()], dtype=float)
    return m if mask is None else m[:, mask]


def bpb_parts(rs: RunSet, rid: str, step: int, set_: str) -> tuple[np.ndarray, np.ndarray]:
    r = rs.row(rid, step, set_, "bpb")
    b = np.array(r["bytes"], dtype=float)
    return np.array(r["per_item"], dtype=float) * b, b  # bits per item, bytes per item


def verdict_dir(est: float, lo: float, hi: float, seeds: list[float], thr_support: float, thr_falsify: float | None = None,
                support_needs_lo: float = 0.0) -> str:
    """SUPPORTED: est >= thr_support, interval lower bound > support_needs_lo, every seed > support_needs_lo.
    FALSIFIED: interval upper bound < thr_falsify (when given). Otherwise UNRESOLVED."""
    if any(not math.isfinite(x) for x in (est, lo, hi)):
        return "UNRESOLVED (non-finite)"
    if est >= thr_support and lo > support_needs_lo and all(s > support_needs_lo for s in seeds):
        return "SUPPORTED"
    if thr_falsify is not None and hi < thr_falsify:
        return "FALSIFIED"
    return "UNRESOLVED"


def fmt(x: float, nd: int = 2) -> str:
    return "nan" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


def write_outputs(report_dir: Path, results: dict, md_lines: list[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (report_dir / "RESULTS_draft.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))
    print(f"\nwrote {report_dir}/results.json, RESULTS_draft.md and the figure")
