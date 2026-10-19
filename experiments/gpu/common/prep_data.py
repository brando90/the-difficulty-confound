"""Build the training pools and evaluation sets for the GPU plans, once, on CPU.

Downloads the public datasets named in suite.json, makes deterministic held-out splits,
removes training documents that share a 13-word sequence with any evaluation item
(decontamination), and writes everything as JSON lines plus a manifest with counts,
SHA-256 hashes and the complexity proxy K for each maths level.

    python prep_data.py --data-root $DATA            # real data (about 6 GB download, 30-60 min on 8 CPUs)
    python prep_data.py --data-root $DATA --toy      # tiny synthetic data for the CPU smoke test (no network)

Output: $DATA/v1/pools/<pool>.jsonl  ({"id", "text"} per training document)
        $DATA/v1/evals/<set>.jsonl   (math: id, problem, answer, level, prompt, target; text sets: id, prompt, target;
                                       hellaswag: id, ctx, endings, label)
        $DATA/v1/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runs  # noqa: E402

WORD = re.compile(r"\w+", re.UNICODE)


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def order_key(seed: int, pool: str, ident: str) -> str:
    return h(f"{seed}:{pool}:{ident}")


def ngrams(text: str, n: int) -> set[int]:
    w = WORD.findall(text.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gsm8k_answer(ans: str) -> tuple[str, str]:
    """GSM8K 'answer' = worked solution with <<calc>> annotations, then '#### N'."""
    sol, _, final = ans.partition("####")
    sol = re.sub(r"<<[^>]*>>", "", sol).strip()
    final = final.strip().replace(",", "")
    return sol + f"\nThe answer is $\\boxed{{{final}}}$.", final


# ---------------------------------------------------------------- real data
def load_real(suite: dict) -> tuple[dict, dict]:
    from datasets import load_dataset

    fmt = suite["format"]
    pools: dict[str, list[dict]] = {}
    evals: dict[str, list[dict]] = {}

    print("loading NuminaMath-CoT ...", flush=True)
    numina = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    src_to_pool = {}
    for name, p in suite["pools"].items():
        for s in p.get("sources", []):
            src_to_pool[s] = name
    by_pool = defaultdict(list)
    for i, (src, prob, sol) in enumerate(zip(numina["source"], numina["problem"], numina["solution"])):
        pool = src_to_pool.get(src)
        if pool and prob and sol:
            by_pool[pool].append({"id": f"numina-{i}", "src": src, "problem": prob.strip(), "solution": sol.strip()})
    pools.update(by_pool)

    print("loading FineWeb-Edu shard ...", flush=True)
    wcfg = suite["pools"]["web"]
    web = load_dataset(wcfg["hf"], data_files=wcfg["data_files"], split="train")
    n = min(wcfg["max_docs"], len(web))
    pools["web"] = [{"id": f"fwe-{web[i]['id']}", "text": web[i]["text"].strip()} for i in range(n) if web[i]["text"].strip()]

    print("loading Gutenberg poetry ...", flush=True)
    pcfg = suite["pools"]["poetry"]
    poem = load_dataset(pcfg["hf"], split=pcfg["split"])
    books = defaultdict(list)
    for line, gid in zip(poem["line"], poem["gutenberg_id"]):
        books[int(gid)].append(line)
    docs = []
    k = pcfg["lines_per_doc"]
    for gid in sorted(books):
        lines = books[gid]
        for j in range(0, len(lines), k):
            chunk = "\n".join(lines[j:j + k]).strip()
            if len(chunk) > 200:
                docs.append({"id": f"gut-{gid}-{j // k}", "book": gid, "text": chunk})
    pools["poetry"] = docs

    print("loading evaluation sets ...", flush=True)
    g = load_dataset("openai/gsm8k", "main", split="test")
    evals["gsm8k"] = []
    for i, r in enumerate(g):
        sol, final = gsm8k_answer(r["answer"])
        evals["gsm8k"].append({"id": f"gsm8k-test-{i}", "problem": r["question"].strip(), "solution": sol, "answer": final, "level": None})
    m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    evals["math500"] = [{"id": r["unique_id"], "problem": r["problem"].strip(), "solution": r["solution"].strip(),
                         "answer": r["answer"].strip(), "level": int(r["level"]), "subject": r["subject"]} for r in m]
    hs = load_dataset("Rowan/hellaswag", split="validation")
    lim = suite["evals"]["hellaswag"]["limit"]
    evals["hellaswag"] = [{"id": f"hs-{r['ind']}", "activity": r["activity_label"], "ctx_a": r["ctx_a"], "ctx_b": r["ctx_b"],
                           "endings": list(r["endings"]), "label": int(r["label"])} for r in list(hs)[:lim]]
    for e in ("gsm8k", "math500"):
        for r in evals[e]:
            r["prompt"] = fmt["math_prompt"].format(problem=r["problem"])
            r["target"] = " " + r["solution"]
    return pools, evals


# ---------------------------------------------------------------- toy data
def load_toy(suite: dict) -> tuple[dict, dict]:
    """Deterministic synthetic stand-ins with the same schema, for the CPU smoke test only."""
    rnd = random.Random(0)
    fmt = suite["format"]

    def arith(level: int, i: int) -> dict:
        a, b = rnd.randint(1, 10 ** level), rnd.randint(1, 10 ** level)
        return {"id": f"toy-L{level}-{i}", "src": f"toy{level}", "problem": f"What is {a} plus {b}?",
                "solution": f"We add {a} and {b} to get {a + b}. The answer is $\\boxed{{{a + b}}}$.", "answer": str(a + b)}

    pools = {
        "L1_gradeschool": [arith(1, i) for i in range(400)],
        "L2_competition": [arith(3, i) for i in range(400)],
        "L3_olympiad": [arith(6, i) for i in range(400)],
        "web": [{"id": f"toy-web-{i}", "text": " ".join(rnd.choice(["the", "cell", "river", "history", "energy", "school"]) for _ in range(60))} for i in range(400)],
        "poetry": [{"id": f"toy-poem-{i}", "book": i // 10, "text": "\n".join(rnd.choice(["O rose thou art sick", "the moon is down", "and I have miles to go", "my heart leaps up"]) for _ in range(8))} for i in range(400)],
    }
    evals = {"gsm8k": [], "math500": []}
    for i in range(30):
        r = arith(1, 10_000 + i)
        evals["gsm8k"].append({"id": f"toy-gsm-{i}", "problem": r["problem"], "solution": r["solution"], "answer": r["answer"], "level": None})
        r = arith(4, 20_000 + i)
        evals["math500"].append({"id": f"toy-math-{i}", "problem": r["problem"], "solution": r["solution"], "answer": r["answer"], "level": 1 + i % 5})
    for e in ("gsm8k", "math500"):
        for r in evals[e]:
            r["prompt"] = fmt["math_prompt"].format(problem=r["problem"])
            r["target"] = " " + r["solution"]
    evals["hellaswag"] = [{"id": f"toy-hs-{i}", "activity": "Toy", "ctx_a": "A man walks to the", "ctx_b": "river and",
                           "endings": ["swims.", "eats a car.", "flies to Mars.", "is a number."], "label": 0} for i in range(20)]
    return pools, evals


# ---------------------------------------------------------------- shared processing
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="tokenizer for the complexity proxy K (median solution tokens)")
    a = ap.parse_args()
    suite = runs.load_suite()
    seed, nheld, ngram_n = suite["split_seed"], suite["heldout_per_pool"], suite["decontam_ngram"]
    root = Path(a.data_root) / suite["data_version"]
    if (root / "manifest.json").exists():
        raise SystemExit(f"{root}/manifest.json exists; data is built once. Move it aside to rebuild.")
    t0 = time.time()
    pools, evals = (load_toy if a.toy else load_real)(suite)
    if a.toy:
        nheld = 20
    fmt = suite["format"]
    manifest: dict = {"data_version": suite["data_version"], "toy": a.toy, "split_seed": seed, "pools": {}, "evals": {}}

    # held-out splits
    train: dict[str, list[dict]] = {}
    for pool, docs in pools.items():
        if pool == "poetry":
            book_order = sorted({d["book"] for d in docs}, key=lambda b: order_key(seed, pool, str(b)))
            held_books, count = set(), 0
            per_book = defaultdict(int)
            for d in docs:
                per_book[d["book"]] += 1
            for b in book_order:
                if count >= nheld:
                    break
                held_books.add(b)
                count += per_book[b]
            held_docs = sorted([d for d in docs if d["book"] in held_books], key=lambda d: order_key(seed, pool, d["id"]))[:nheld]
            train[pool] = [d for d in docs if d["book"] not in held_books]
            manifest["pools"][pool] = {"heldout_books": len(held_books)}
        else:
            ordered = sorted(docs, key=lambda d: order_key(seed, pool, d["id"]))
            held_docs, train[pool] = ordered[:nheld], ordered[nheld:]
            manifest["pools"][pool] = {}
        name = {"L1_gradeschool": "L1_heldout", "L2_competition": "L2_heldout", "L3_olympiad": "L3_heldout",
                "web": "web_heldout", "poetry": "poetry_heldout"}[pool]
        rows = []
        for d in held_docs:
            if "problem" in d:
                rows.append({"id": d["id"], "problem": d["problem"], "solution": d["solution"], "level": pool[:2],
                             "prompt": fmt["math_prompt"].format(problem=d["problem"]), "target": " " + d["solution"]})
            else:
                rows.append({"id": d["id"], "prompt": "", "target": d["text"][:6000]})
        evals[name] = rows

    # decontamination against every maths evaluation item and held-out document
    banned: set[int] = set()
    for name in ("gsm8k", "math500", "L1_heldout", "L2_heldout", "L3_heldout"):
        for r in evals[name]:
            banned |= ngrams(r["problem"] + " " + r.get("solution", ""), ngram_n)
    held_text_hash = {h(r["target"]) for name in ("web_heldout", "poetry_heldout") for r in evals[name]}
    for pool, docs in train.items():
        kept, removed = [], 0
        for d in docs:
            text = fmt["math_doc"].format(problem=d["problem"], solution=d["solution"]) if "problem" in d else d["text"]
            if "problem" in d:
                if ngrams(d["problem"] + " " + d["solution"], ngram_n) & banned:
                    removed += 1
                    continue
            elif h(text[:6000]) in held_text_hash:
                removed += 1
                continue
            kept.append({"id": d["id"], "text": text})
        sha = write_jsonl(root / "pools" / f"{pool}.jsonl", kept)
        manifest["pools"][pool].update({"train_docs": len(kept), "removed_by_decontam": removed, "sha256": sha,
                                        "train_chars": sum(len(d["text"]) for d in kept),
                                        "licence": suite["pools"][pool]["licence"]})
        print(f"pool {pool}: {len(kept)} train docs, {removed} removed by decontamination", flush=True)

    # complexity proxy K per maths level: median solution length in tokens on the held-out targets
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(suite["models"]["tiny-random"]["hf"] if a.toy else a.tokenizer)
        count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
        tokname = tok.name_or_path
    except Exception as exc:  # pragma: no cover - only if transformers is missing
        print("tokenizer unavailable, K uses words:", exc)
        count = lambda s: len(WORD.findall(s))  # noqa: E731
        tokname = "words"
    manifest["complexity_K"] = {"proxy": f"median solution length in tokens ({tokname}) over the 1,000 held-out targets", "levels": {}}
    for lvl in ("L1", "L2", "L3"):
        lens = [count(r["target"]) for r in evals[f"{lvl}_heldout"]]
        manifest["complexity_K"]["levels"][lvl] = {"median_tokens": statistics.median(lens), "mean_tokens": statistics.fmean(lens)}

    for name, rows in evals.items():
        sha = write_jsonl(root / "evals" / f"{name}.jsonl", rows)
        manifest["evals"][name] = {"n": len(rows), "sha256": sha, "licence": suite["evals"].get(name, {}).get("licence", "see source pool")}
    manifest["build_seconds"] = round(time.time() - t0, 1)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["complexity_K"], indent=2))
    print(f"wrote {root} in {manifest['build_seconds']} s")


if __name__ == "__main__":
    main()
