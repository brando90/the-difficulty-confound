"""Train one run of the shared registry (or evaluate a base model) and write its metrics.

Full fine-tuning of a pretrained causal language model on a token-exact mixture of pools:
documents are drawn so that after every document each pool's share of the tokens seen is within
one document of its target fraction, separated by EOS, and packed into seq_len rows. The learning
rate warms up linearly for `warmup_steps` and then stays constant, so every checkpoint is a
comparable stopping point. At each checkpoint the run scores all bits-per-byte sets; at generation
checkpoints it also decodes GSM8K and MATH-500 and scores HellaSwag.

    python train.py --tier main --plans g1,g2,g3 --index 7 --data-root $DATA --out-root $OUT
    python train.py --tier main --plans g1,g2,g3 --run-id <id> --data-root $DATA --out-root $OUT
    python train.py ... --check-only     # load data and model, run 2 steps and every evaluation on 8 items

Writes $OUT/runs/<run_id>/{spec.json, metrics.jsonl, train_log.jsonl, generations/, summary.json, DONE}.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evals  # noqa: E402
import runs  # noqa: E402

BPB_SETS = ["L1_heldout", "L2_heldout", "L3_heldout", "gsm8k", "math500", "poetry_heldout", "web_heldout"]
GEN_SETS = ["gsm8k", "math500"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]


def answer_only(text: str) -> str | None:
    """Format-only document: the same problem, with its worked solution replaced by the bare final answer in the
    answer convention the evaluations grade ("The answer is $\\boxed{...}$."). None when the solution has no \\boxed{}."""
    problem, sep, solution = text.partition("\nSolution:")
    ans = evals.last_boxed(solution) if sep else None
    if not problem.startswith("Problem:") or ans is None or not ans.strip():
        return None
    return f"{problem}\nSolution: The answer is $\\boxed{{{ans.strip()}}}$."


TRANSFORMS = {"answer_only": answer_only}


class MixtureStream:
    """Deterministic token stream over several pools with exact token shares. `transform` (optional, a TRANSFORMS
    name) rewrites every document of every pool when the pools are loaded; documents it returns None for are dropped."""

    def __init__(self, data_root: Path, mixture: dict, caps: dict, tok, seed: int, seq_len: int, transform: str | None = None):
        self.tok, self.seq_len, self.eos = tok, seq_len, tok.eos_token_id
        self.pools, self.frac, self.order, self.pos, self.epochs = {}, {}, {}, {}, {}
        self.drawn = {p: 0 for p in mixture}
        self.rng = random.Random(seed)
        self.dropped = {}
        for p, f in mixture.items():
            docs = read_jsonl(data_root / "pools" / f"{p}.jsonl")
            if transform:
                fn, n0 = TRANSFORMS[transform], len(docs)
                docs = [{**d, "text": t} for d in docs if (t := fn(d["text"])) is not None]
                self.dropped[p] = n0 - len(docs)
            order = list(range(len(docs)))
            random.Random(f"{seed}:{p}").shuffle(order)
            if p in caps:  # keep a fixed prefix of the shuffled pool worth `cap` tokens; it is then repeated
                kept, n = [], 0
                for i in order:
                    if n >= caps[p]:
                        break
                    kept.append(i)
                    n += len(tok(docs[i]["text"], add_special_tokens=False)["input_ids"]) + 1
                order = kept
            self.pools[p], self.frac[p], self.order[p], self.pos[p], self.epochs[p] = docs, f, order, 0, 0
        self.unique_docs = {p: len(self.order[p]) for p in self.pools}
        self.buf: list[int] = []
        self.total = 0

    def _next_doc(self) -> list[int]:
        # the pool furthest below its target share draws next
        p = max(self.pools, key=lambda q: self.frac[q] * (self.total + 1) - self.drawn[q])
        if self.pos[p] >= len(self.order[p]):
            self.pos[p] = 0
            self.epochs[p] += 1
            random.Random(f"{self.rng.random()}:{p}:{self.epochs[p]}").shuffle(self.order[p])
        doc = self.pools[p][self.order[p][self.pos[p]]]["text"]
        self.pos[p] += 1
        ids = self.tok(doc, add_special_tokens=False)["input_ids"] + [self.eos]
        self.drawn[p] += len(ids)
        self.total += len(ids)
        return ids

    def batch(self, n: int) -> torch.Tensor:
        need = n * self.seq_len
        while len(self.buf) < need:
            self.buf.extend(self._next_doc())
        out, self.buf = self.buf[:need], self.buf[need:]
        return torch.tensor(out, dtype=torch.long).view(n, self.seq_len)

    def shares(self) -> dict:
        return {p: self.drawn[p] / max(self.total, 1) for p in self.pools}


def load_evals(data_root: Path, limit: int | None) -> dict:
    out = {}
    for name in set(BPB_SETS) | set(GEN_SETS) | {"hellaswag"}:
        rows = read_jsonl(data_root / "evals" / f"{name}.jsonl")
        out[name] = rows[:limit] if limit else rows
    return out


def evaluate(model, tok, ev: dict, spec: dict, step: int, tokens: int, gen: bool, out_dir: Path, device, amp, check: bool) -> list[dict]:
    model.eval()
    rows = []
    bt = 2048 if device.type == "cpu" else 8192
    for name in BPB_SETS:
        t0 = time.time()
        r = evals.bits_per_byte(model, tok, ev[name], spec["seq_len"], bt, device, amp)
        rows.append({"step": step, "tokens": tokens, "set": name, "kind": "bpb", "seconds": round(time.time() - t0, 1), **r})
    if gen:
        g = spec.get("gen") or {"engine": "pilot"}
        for name in GEN_SETS:
            t0 = time.time()
            if g["engine"] == "fast":  # common/evals.py generate_math_fast; same prompts, cap, stop string and graders
                mb = 4 if device.type == "cpu" else g["batch_size"]
                r = evals.generate_math_fast(model, tok, ev[name], spec["gen_max_new_tokens"], mb, device, amp,
                                             spec["format"]["stop_strings"], chunk_tokens=g["chunk_tokens"],
                                             max_batch_tokens=None if device.type == "cpu" else g.get("max_batch_tokens"))
            else:  # the pilot's grader: rounds of 64 prompts (0.5B), each run to the cap when one row rambles
                mb = 4 if device.type == "cpu" else 8 * spec["micro_batch"]  # 64 prompts at 0.5B, 32 at 1.5B
                r = evals.generate_math(model, tok, ev[name], spec["gen_max_new_tokens"], mb, device, amp, spec["format"]["stop_strings"])
            gdir = out_dir / "generations"
            gdir.mkdir(exist_ok=True)
            with (gdir / f"step{step:05d}_{name}.jsonl").open("w") as f:
                for x in r.pop("rows"):
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            rows.append({"step": step, "tokens": tokens, "set": name, "kind": "acc", "seconds": round(time.time() - t0, 1), **r})
        t0 = time.time()
        r = evals.hellaswag(model, tok, ev["hellaswag"], 4 if device.type == "cpu" else 32, device, amp)
        rows.append({"step": step, "tokens": tokens, "set": "hellaswag", "kind": "mc", "seconds": round(time.time() - t0, 1), **r})
    model.train()
    for r in rows:
        msg = {k: r[k] for k in ("step", "set", "kind") if k in r}
        msg.update({k: round(r[k], 4) for k in ("bpb", "acc", "acc_fallback", "acc_norm", "stopped_rate", "boxed_rate", "mean_tokens") if k in r})
        print(json.dumps(msg), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--plans", default="all")
    ap.add_argument("--secondary", action="store_true")
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "all"))
    ap.add_argument("--index", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--save-final", action="store_true", help="also save the final weights (about 1 GB at 0.5B in bf16)")
    a = ap.parse_args()

    suite = runs.load_suite()
    allrun = runs.expand(a.tier, runs.parse_plans(a.plans), a.secondary, allow_unset_lr=a.check_only, seeds=runs.parse_seeds(a.seeds))
    if a.run_id:
        match = [s for s in allrun if s["run_id"] == a.run_id]
        if not match:
            raise SystemExit(f"run id {a.run_id} is not in tier {a.tier} for plans {a.plans}")
        spec = match[0]
    elif a.index is not None:
        if not 0 <= a.index < len(allrun):
            raise SystemExit(f"index {a.index} outside 0..{len(allrun) - 1}")
        spec = allrun[a.index]
    else:
        raise SystemExit("give --index or --run-id")
    if a.check_only and spec["kind"] == "train" and not math.isfinite(spec["lr"]):
        spec["lr"] = 1e-5

    data_root = Path(a.data_root) / suite["data_version"]
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest["toy"] and a.tier != "smoke":
        raise SystemExit("toy data may only be used with --tier smoke")
    out_dir = runs.run_dir(a.out_root, spec["run_id"]) if not a.check_only else Path(a.out_root) / "check_only" / spec["run_id"]
    if (out_dir / "DONE").exists():
        done_spec = json.loads((out_dir / "spec.json").read_text())
        lacking = sorted(set(spec["gen_steps"]) - set(done_spec["gen_steps"]))
        if lacking:  # same run id, but now needed by an arm that wants more evaluations
            raise SystemExit(f"{spec['run_id']} is DONE without the generation evaluations at steps {lacking}; "
                             "move it aside, rerun it, and note it in CKPT.md")
        print(f"{spec['run_id']} already DONE")
        return
    if not a.check_only:  # one claim per run id across every launcher and node; a crashed run keeps its claim (no silent retries)
        claims = Path(a.out_root) / "claims"
        claims.mkdir(parents=True, exist_ok=True)
        try:
            (claims / spec["run_id"]).mkdir()
        except FileExistsError:
            owner = (claims / spec["run_id"] / "owner").read_text().strip() if (claims / spec["run_id"] / "owner").exists() else "unknown"
            print(f"{spec['run_id']} is claimed by {owner} but not DONE; skipping. If that job died: move {out_dir} aside, "
                  f"rm -r {claims / spec['run_id']}, note it in CKPT.md, resubmit.")
            return
        (claims / spec["run_id"] / "owner").write_text(
            f"{platform.node()} pid {os.getpid()} slurm {os.environ.get('SLURM_JOB_ID', '-')} {time.strftime('%m-%d-%Y %H:%M:%S')}\n")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} has partial output from an earlier attempt; move it aside first (runs restart from scratch)")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = spec["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    want = spec["device"]
    device = torch.device("cuda" if want == "cuda" and torch.cuda.is_available() else "cpu")
    if want == "cuda" and device.type != "cuda" and not a.check_only:
        raise SystemExit("this tier needs a GPU and none is visible")
    use_bf16 = device.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    if use_bf16:
        torch.backends.cuda.matmul.allow_tf32 = True

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["hf"], torch_dtype=torch.float32).to(device)
    if device.type == "cuda":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.generation_config.use_cache = True

    limit = spec.get("eval_limit") or (8 if a.check_only else None)
    ev = load_evals(data_root, limit)
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    metrics_f = (out_dir / "metrics.jsonl").open("a")
    log_f = (out_dir / "train_log.jsonl").open("a")
    t_start = time.time()

    def record(rows: list[dict]) -> None:
        for r in rows:
            metrics_f.write(json.dumps(r) + "\n")
        metrics_f.flush()

    summary = {"run_id": spec["run_id"], "host": platform.node(), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               "data_manifest_sha256": hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest()}
    if spec["kind"] == "base":
        record(evaluate(model, tok, ev, spec, 0, 0, True, out_dir, device, amp, a.check_only))
    else:
        tokens_per_step = spec["seqs_per_step"] * spec["seq_len"]
        mb = min(spec["micro_batch"], spec["seqs_per_step"])
        accum = spec["seqs_per_step"] // mb
        assert accum * mb == spec["seqs_per_step"], "seqs_per_step must be a multiple of micro_batch"
        stream = MixtureStream(data_root, spec["mixture"], spec["caps"], tok, seed, spec["seq_len"], spec.get("transform"))
        o = spec["optim"]
        opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], betas=tuple(o["betas"]), weight_decay=o["weight_decay"],
                                fused=device.type == "cuda")
        max_steps = 2 if a.check_only else spec["max_steps"]
        ckpts = {2} if a.check_only else set(spec["ckpt_steps"])
        gens = {2} if a.check_only else set(spec["gen_steps"])
        model.train()
        t_train = 0.0
        for step in range(1, max_steps + 1):
            t0 = time.time()
            lr = spec["lr"] * min(1.0, step / max(o["warmup_steps"], 1))
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            for _ in range(accum):
                x = stream.batch(mb).to(device, non_blocking=True)
                with amp():
                    loss = model(input_ids=x, labels=x).loss
                (loss / accum).backward()
                loss_sum += loss.item() / accum
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"]).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t_train += dt
            if not math.isfinite(loss_sum):
                (out_dir / "FAILED").write_text(f"non-finite loss at step {step}\n")
                raise SystemExit(f"non-finite loss at step {step}; run marked FAILED (never retried silently)")
            log_f.write(json.dumps({"step": step, "tokens": step * tokens_per_step, "loss": round(loss_sum, 5), "grad_norm": round(gnorm, 4),
                                    "lr": lr, "seconds": round(dt, 3), "shares": {k: round(v, 5) for k, v in stream.shares().items()}}) + "\n")
            if step % 10 == 0 or step == 1:
                log_f.flush()
                print(f"step {step}/{max_steps} loss {loss_sum:.4f} tok/s {tokens_per_step / dt:,.0f}", flush=True)
            if step in ckpts:
                record(evaluate(model, tok, ev, spec, step, step * tokens_per_step, step in gens, out_dir, device, amp, a.check_only))
        summary.update({"train_seconds": round(t_train, 1), "tokens": max_steps * tokens_per_step,
                        "tokens_per_second": round(max_steps * tokens_per_step / max(t_train, 1e-9)),
                        "realized_shares": stream.shares(), "epochs_completed": stream.epochs, "unique_docs": stream.unique_docs,
                        "transform": spec.get("transform"), "dropped_by_transform": stream.dropped,
                        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None})
        if a.save_final and not a.check_only:
            model.to(torch.bfloat16).save_pretrained(out_dir / "final", safe_serialization=True)
            tok.save_pretrained(out_dir / "final")
    metrics_f.close()
    log_f.close()
    summary["wall_seconds"] = round(time.time() - t_start, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sha = hashlib.sha256((out_dir / "metrics.jsonl").read_bytes()).hexdigest()
    (out_dir / ("CHECK_OK" if a.check_only else "DONE")).write_text(f"{sha}  metrics.jsonl\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
