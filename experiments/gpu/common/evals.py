"""Evaluations shared by the GPU plans.

bits_per_byte  code length of each target text given its prompt, in bits per UTF-8 byte of the target
               (tokenizer-independent, so models and checkpoints compare directly)
generate_math  greedy decoding of "Problem: ...\\nSolution:" prompts, graded with Math-Verify against
               the gold answer, plus an independent last-\\boxed{}/last-number grader for agreement
hellaswag      4-way multiple choice by length-normalised log-likelihood (acc_norm)
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def _pad_right(seqs: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


def _token_nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """nll[r, k] = -log p(ids[r, k+1] | ids[r, :k+1]) in nats, one row at a time (a full-batch float32
    log-softmax over a 152k vocabulary would need tens of GB)."""
    out = torch.empty(ids.shape[0], ids.shape[1] - 1, device=ids.device, dtype=torch.float32)
    for r in range(ids.shape[0]):
        out[r] = F.cross_entropy(logits[r, :-1].float(), ids[r, 1:], reduction="none")
    return out


@torch.no_grad()
def bits_per_byte(model, tok, items: list[dict], seq_len: int, batch_tokens: int, device, amp) -> dict:
    """items: {"id", "prompt", "target"}. Every sequence starts with EOS, as documents do in training."""
    eos = tok.eos_token_id
    enc = []
    for it in items:
        p = tok(it["prompt"], add_special_tokens=False)["input_ids"] if it["prompt"] else []
        t = tok(it["target"], add_special_tokens=False)["input_ids"]
        p = [eos] + p
        room = seq_len - len(p)
        if room < 8:  # very long prompt: keep its end
            p = [eos] + p[-(seq_len // 2):]
            room = seq_len - len(p)
        t = t[:room]
        nbytes = len(tok.decode(t).encode("utf-8"))
        enc.append((p, t, nbytes))
    order = sorted(range(len(enc)), key=lambda i: -(len(enc[i][0]) + len(enc[i][1])))
    per_item = [0.0] * len(enc)
    i = 0
    while i < len(order):
        L = len(enc[order[i]][0]) + len(enc[order[i]][1])
        bs = max(1, batch_tokens // max(L, 1))
        idx = order[i : i + bs]
        i += bs
        seqs = [enc[j][0] + enc[j][1] for j in idx]
        ids, mask = _pad_right(seqs, eos)
        ids, mask = ids.to(device), mask.to(device)
        with amp():
            logits = model(input_ids=ids, attention_mask=mask).logits
        nll = _token_nll(logits, ids)
        del logits
        for row, j in enumerate(idx):
            plen, tlen = len(enc[j][0]), len(enc[j][1])
            # target token k sits at position plen + k and is predicted from position plen + k - 1
            s = nll[row, plen - 1 : plen - 1 + tlen].sum().item()
            per_item[j] = s / LN2 / max(enc[j][2], 1)
    total_bits = sum(per_item[j] * max(enc[j][2], 1) for j in range(len(enc)))
    total_bytes = sum(max(e[2], 1) for e in enc)
    return {"bpb": total_bits / total_bytes, "n": len(enc), "per_item": [round(x, 6) for x in per_item],
            "bytes": [max(e[2], 1) for e in enc]}


# ---------------------------------------------------------------- answer grading
def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def _norm(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(left|right|!|,|;|:)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").replace(",", "") if re.fullmatch(r"[-\d,\. ]+", s) else s.replace(" ", "")
    s = s.rstrip(".")
    try:
        f = float(s)
        return f"{f:g}"
    except ValueError:
        return s


def fallback_grade(generation: str, gold: str) -> bool:
    pred = last_boxed(generation)
    if pred is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", generation)
        pred = nums[-1] if nums else None
    return pred is not None and _norm(pred) == _norm(gold)


def mv_grade(generation: str, gold: str) -> bool | None:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        g = parse(f"${gold}$")
        p = parse(generation)
        return bool(verify(g, p))
    except Exception:
        return False


@torch.no_grad()
def generate_math(model, tok, items: list[dict], max_new_tokens: int, batch_size: int, device, amp, stop_strings) -> dict:
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["prompt"]))
    gens = [""] * len(items)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        prompts = [items[i]["prompt"] for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with amp():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                 stop_strings=list(stop_strings), tokenizer=tok)
        new = out[:, enc["input_ids"].shape[1]:]
        for row, i in enumerate(idx):
            text = tok.decode(new[row], skip_special_tokens=True)
            for s in stop_strings:
                text = text.split(s)[0]
            gens[i] = text
    correct, correct_fb, rows = [], [], []
    for it, g in zip(items, gens):
        mv = mv_grade(g, it["answer"])
        fb = fallback_grade(g, it["answer"])
        c = fb if mv is None else mv
        correct.append(int(c))
        correct_fb.append(int(fb))
        rows.append({"id": it["id"], "answer": it["answer"], "level": it.get("level"), "correct": int(c), "correct_fallback": int(fb), "generation": g})
    n = len(items)
    agree = sum(int(x == y) for x, y in zip(correct, correct_fb)) / max(n, 1)
    return {"acc": sum(correct) / max(n, 1), "acc_fallback": sum(correct_fb) / max(n, 1), "grader_agreement": agree,
            "n": n, "correct": correct, "levels": [it.get("level") for it in items], "rows": rows}


@torch.no_grad()
def generate_math_fast(model, tok, items: list[dict], max_new_tokens: int, batch_size: int, device, amp, stop_strings,
                       chunk_tokens: int = 256, max_batch_tokens: int | None = None) -> dict:
    """generate_math at a fraction of the cost, with the same prompts, greedy decoding, stop string, token cap and
    graders. The pilot's cost was rounds of 64 prompts that each ran to the full cap whenever one row rambled. Here
    batches are large, and decoding goes in rounds of `chunk_tokens`: a row that emitted EOS, hit the stop string or
    reached the cap leaves; the rest continue in the next round from their own prompt + generated token ids (re-encoded
    as the prompt, never re-tokenised text), regrouped by length. Greedy continuation of the same token ids is the same
    computation, so outputs match generate_math up to floating-point differences from batch composition.
    `max_batch_tokens` caps rows x (prompt + generated + budget) per batch, so later rounds of longer rows use
    smaller batches.
    Also records per item: generated tokens, whether it stopped by itself (EOS or stop string) and whether it gave a
    \\boxed{} answer."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad, eos = tok.pad_token_id, tok.eos_token_id
    end_ids = {pad, eos}
    prompt_ids = [tok(it["prompt"], add_special_tokens=False)["input_ids"] for it in items]
    gen_ids: list[list[int]] = [[] for _ in items]
    stopped = [False] * len(items)
    active, rounds, batches = list(range(len(items))), 0, 0
    while active:
        rounds += 1
        budget = min(chunk_tokens, max_new_tokens - len(gen_ids[active[0]]))  # every active row has the same length
        active.sort(key=lambda i: -(len(prompt_ids[i]) + len(gen_ids[i])))
        nxt, b = [], 0
        while b < len(active):
            width = len(prompt_ids[active[b]]) + len(gen_ids[active[b]]) + budget  # the longest row of this batch
            rows_ = batch_size if not max_batch_tokens else max(1, min(batch_size, max_batch_tokens // width))
            idx = active[b : b + rows_]
            b += rows_
            batches += 1
            seqs = [prompt_ids[i] + gen_ids[i] for i in idx]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, s in enumerate(seqs):
                ids[row, width - len(s):] = torch.tensor(s)
                mask[row, width - len(s):] = 1
            with amp():
                out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=budget,
                                     do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=pad,
                                     eos_token_id=eos, stop_strings=list(stop_strings), tokenizer=tok)
            new = out[:, width:].tolist()
            for row, i in enumerate(idx):
                toks = new[row]
                cut = next((k for k, t in enumerate(toks) if t in end_ids), None)
                gen_ids[i] += toks if cut is None else toks[:cut]
                if cut is not None or any(s in tok.decode(gen_ids[i], skip_special_tokens=True) for s in stop_strings):
                    stopped[i] = True
                elif len(gen_ids[i]) < max_new_tokens:
                    nxt.append(i)
        active = nxt
    gens = []
    for g in gen_ids:
        text = tok.decode(g, skip_special_tokens=True)
        for s in stop_strings:
            text = text.split(s)[0]
        gens.append(text)
    correct, correct_fb, rows = [], [], []
    for it, g, ids_, st in zip(items, gens, gen_ids, stopped):
        mv = mv_grade(g, it["answer"])
        fb = fallback_grade(g, it["answer"])
        c = fb if mv is None else mv
        correct.append(int(c))
        correct_fb.append(int(fb))
        rows.append({"id": it["id"], "answer": it["answer"], "level": it.get("level"), "correct": int(c), "correct_fallback": int(fb),
                     "n_tokens": len(ids_), "stopped": int(st), "boxed": int(last_boxed(g) is not None), "generation": g})
    n = len(items)
    agree = sum(int(x == y) for x, y in zip(correct, correct_fb)) / max(n, 1)
    return {"acc": sum(correct) / max(n, 1), "acc_fallback": sum(correct_fb) / max(n, 1), "grader_agreement": agree,
            "n": n, "correct": correct, "levels": [it.get("level") for it in items],
            "stopped_rate": sum(stopped) / max(n, 1), "boxed_rate": sum(r["boxed"] for r in rows) / max(n, 1),
            "mean_tokens": sum(len(g) for g in gen_ids) / max(n, 1), "engine": {"name": "fast", "batch_size": batch_size,
            "chunk_tokens": chunk_tokens, "max_batch_tokens": max_batch_tokens, "max_new_tokens": max_new_tokens,
            "rounds": rounds, "batches": batches}, "rows": rows}


@torch.no_grad()
def hellaswag(model, tok, items: list[dict], batch_size: int, device, amp) -> dict:
    eos = tok.eos_token_id

    def prep(t: str) -> str:
        t = t.strip().replace(" [title]", ". ")
        t = re.sub(r"\[.*?\]", "", t)
        return t.replace("  ", " ")

    correct = []
    for b in range(0, len(items), batch_size):
        chunk = items[b : b + batch_size]
        seqs, spans, lens = [], [], []
        for it in chunk:
            ctx = prep(it["activity"] + ": " + it["ctx_a"] + " " + it["ctx_b"].capitalize())
            c = [eos] + tok(ctx, add_special_tokens=False)["input_ids"]
            for e in it["endings"]:
                ending = " " + prep(e)
                t = tok(ending, add_special_tokens=False)["input_ids"]
                seqs.append(c + t)
                spans.append((len(c), len(t)))
                lens.append(len(ending))
        ids, mask = _pad_right(seqs, eos)
        ids, mask = ids.to(device), mask.to(device)
        with amp():
            logits = model(input_ids=ids, attention_mask=mask).logits
        ll = -_token_nll(logits, ids)
        del logits
        k = 0
        for it in chunk:
            scores = []
            for _ in it["endings"]:
                start, n = spans[k]
                scores.append(ll[k, start - 1 : start - 1 + n].sum().item() / lens[k])
                k += 1
            correct.append(int(max(range(len(scores)), key=scores.__getitem__) == it["label"]))
    return {"acc_norm": sum(correct) / max(len(correct), 1), "n": len(correct), "correct": correct}
