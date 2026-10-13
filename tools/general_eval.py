"""Small general-ability suite for checking that an auxiliary loss does not hurt the model outside arithmetic:
  hellaswag      validation, 4-way, length-normalised log-likelihood of the ending (acc_norm-style)
  arc_easy       test, multiple choice, "Question: q\\nAnswer:" + " " + choice, length-normalised log-likelihood
  arc_challenge  test, same format
  lambada        OpenAI LAMBADA (en), greedy accuracy on the last word
n examples per task (fixed first n), the same items for every model. Data: HF parquet / jsonl files in --dir.

    from general_eval import load_suite, run_suite
    suite = load_suite("/scratch/.../data/general", n=500); scores = run_suite(model, tok, suite, dev)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn


def load_suite(root: str, n: int = 500) -> dict:
    import pandas as pd
    root = Path(root)
    hs = pd.read_parquet(root / "hellaswag/data/validation-00000-of-00001.parquet").iloc[:n]
    suite = {"hellaswag": [(f"{r.activity_label}: {r.ctx}", [" " + e for e in r.endings], int(r.label)) for r in hs.itertuples()]}
    for name, sub in (("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")):
        df = pd.read_parquet(root / f"ai2_arc/{sub}/test-00000-of-00001.parquet").iloc[:n]
        items = []
        for r in df.itertuples():
            labels = list(r.choices["label"])
            if r.answerKey not in labels:
                continue
            items.append((f"Question: {r.question}\nAnswer:", [" " + t for t in r.choices["text"]], labels.index(r.answerKey)))
        suite[name] = items
    lam = [json.loads(x)["text"] for x in open(root / "lambada_openai/data/lambada_test_en.jsonl")][:n]
    suite["lambada"] = [(t.rsplit(" ", 1)[0], " " + t.rsplit(" ", 1)[1]) for t in lam]
    return suite


@torch.no_grad()
def _logliks(model, tok, ctx: str, conts: list, dev) -> np.ndarray:
    """Mean per-token log-probability of each continuation given ctx (one right-padded batch)."""
    c = tok(ctx, add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None:
        c = [tok.bos_token_id] + c
    seqs = [c + tok(x, add_special_tokens=False)["input_ids"] for x in conts]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long, device=dev)
    att = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, device=dev); att[i, :len(s)] = 1
    with torch.autocast("cuda" if str(dev).startswith("cuda") else "cpu", dtype=torch.bfloat16):
        lp = Fn.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
    out = []
    for i, s in enumerate(seqs):
        pos = torch.arange(len(c) - 1, len(s) - 1, device=dev)
        out.append(float(lp[i, pos, ids[i, pos + 1]].mean()))
    return np.array(out)


@torch.no_grad()
def run_suite(model, tok, suite: dict, dev) -> dict:
    was_training = model.training
    model.eval()
    res = {}
    for name in ("hellaswag", "arc_easy", "arc_challenge"):
        hit = [int(np.argmax(_logliks(model, tok, ctx, conts, dev)) == y) for ctx, conts, y in suite[name]]
        res[name] = float(np.mean(hit))
    hit = []
    for ctx, word in suite["lambada"]:
        c = tok(ctx, add_special_tokens=False)["input_ids"]
        if tok.bos_token_id is not None:
            c = [tok.bos_token_id] + c
        w = tok(word, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([c + w], device=dev)
        with torch.autocast("cuda" if str(dev).startswith("cuda") else "cpu", dtype=torch.bfloat16):
            pred = model(input_ids=ids).logits[0, len(c) - 1:len(c) - 1 + len(w)].argmax(-1)
        hit.append(int(pred.tolist() == w))
    res["lambada"] = float(np.mean(hit))
    res["mean"] = float(np.mean([res[k] for k in ("hellaswag", "arc_easy", "arc_challenge", "lambada")]))
    if was_training:
        model.train()
    return res
