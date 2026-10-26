"""Exact-equivalence check of the decoding algorithm: generate_math (pilot) vs generate_math_fast, token for token,
on the real GSM8K / MATH-500 prompts, with a tiny random Qwen2 (fast on CPU). Several stop-string settings so that rows
stop inside rounds, at round boundaries and never; the 768 cap is replaced by a small one."""
import json, sys, time, contextlib
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "../../../common"))
import torch, evals
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(2)
P = "Problem: {problem}\nSolution:"
g = load_dataset("openai/gsm8k", "main", split="test"); m = load_dataset("HuggingFaceH4/MATH-500", split="test")
items = [{"id": f"gsm8k-test-{i}", "prompt": P.format(problem=g[i]["question"].strip()), "answer": g[i]["answer"].split("####")[1].strip(), "level": None} for i in range(0, 1319, 11)]
items += [{"id": m[i]["unique_id"], "prompt": P.format(problem=m[i]["problem"].strip()), "answer": m[i]["answer"].strip(), "level": int(m[i]["level"])} for i in range(0, 500, 10)]
name = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
tok.pad_token = tok.pad_token or tok.eos_token
amp, dev = contextlib.nullcontext, torch.device("cpu")
out = []
for stop, cap, chunk, mbt in ((["\nProblem:"], 40, 16, None), ([" $\\"], 40, 16, None), (["{"], 48, 12, 4000), (["\nProblem:", "The"], 33, 10, 3000)):
    t0 = time.time(); old = evals.generate_math(model, tok, items, cap, 8, dev, amp, stop); t1 = time.time()
    new = evals.generate_math_fast(model, tok, items, cap, 32, dev, amp, stop, chunk_tokens=chunk, max_batch_tokens=mbt); t2 = time.time()
    same = sum(x["generation"] == y["generation"] for x, y in zip(old["rows"], new["rows"]))
    r = {"stop": stop, "cap": cap, "chunk": chunk, "max_batch_tokens": mbt, "n": len(items), "same_generation": same,
         "same_correct": sum(x == y for x, y in zip(old["correct"], new["correct"])), "stopped_rate": round(new["stopped_rate"], 3),
         "mean_tokens": round(new["mean_tokens"], 1), "engine": new["engine"], "sec_old": round(t1 - t0, 1), "sec_new": round(t2 - t1, 1),
         "first_diff": next(((x["id"], x["generation"][:120], y["generation"][:120]) for x, y in zip(old["rows"], new["rows"]) if x["generation"] != y["generation"]), None)}
    print(json.dumps(r), flush=True); out.append(r)
json.dump(out, open("equiv_tiny.json", "w"), indent=1)
