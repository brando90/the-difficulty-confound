#!/usr/bin/env python3
"""Re-grade a seeds3 run from its saved final weights (train.py --save-final), without retraining.

    python regrade.py --run-dir $OUT/runs/<run_id> --data-root $DATA --engine pilot

Writes <run-dir>/regrade_<engine>/metrics.jsonl and generations, in the same format as train.py. Use it when the
pre-registered grader check fails (CKPT.md): every arm is then re-graded with the pilot's engine before any readout.
The base run has no saved weights; it loads the model from the hub cache.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import evals  # noqa: E402
import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--engine", choices=["pilot", "fast"], default="pilot")
    a = ap.parse_args()
    run = Path(a.run_dir)
    spec = json.loads((run / "spec.json").read_text())
    weights = run / "final" if (run / "final").exists() else spec["hf"]
    if spec["kind"] == "train" and weights == spec["hf"]:
        raise SystemExit(f"{run} has no final/ weights (it was not run with --save-final)")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(spec["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32).to(device).eval()
    model.generation_config.use_cache = True
    torch.backends.cuda.matmul.allow_tf32 = True
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    suite_root = Path(a.data_root) / "v1"
    ev = train.load_evals(suite_root, spec.get("eval_limit"))
    spec["gen"] = {**(spec.get("gen") or {}), "engine": a.engine}
    out = run / f"regrade_{a.engine}"
    out.mkdir(exist_ok=True)
    rows = []
    for name in train.GEN_SETS:
        if a.engine == "fast":
            g = spec["gen"]
            r = evals.generate_math_fast(model, tok, ev[name], spec["gen_max_new_tokens"], g["batch_size"], device, amp,
                                         spec["format"]["stop_strings"], chunk_tokens=g["chunk_tokens"], max_batch_tokens=g.get("max_batch_tokens"))
        else:
            r = evals.generate_math(model, tok, ev[name], spec["gen_max_new_tokens"], 8 * spec["micro_batch"], device, amp,
                                    spec["format"]["stop_strings"])
        with (out / f"{name}.jsonl").open("w") as f:
            for x in r.pop("rows"):
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        rows.append({"step": max(spec["gen_steps"]), "set": name, "kind": "acc", **r})
        print(name, round(r["acc"], 4), flush=True)
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
