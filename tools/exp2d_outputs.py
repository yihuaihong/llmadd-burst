"""Idea 2, follow-up: what does the model SAY on a+b+c= when its '=' state already encodes a+b+c?

At the end of stage 1 the total is decodable from the '=' state for 75% of unseen (a,b) pairs while
first-token accuracy is 10%. Before calling that a know-but-can't-say gap, rule out format: greedy-
generate a few tokens and classify the first token and the first integer of the continuation, and
report the probability and rank of the correct first token.

    python tools/exp2d_outputs.py --model <dir> --out <dir> [--n 400]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import numlib as nl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--new_tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    rng = np.random.default_rng(args.seed)
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    pool = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 99])
    sel = pool[rng.permutation(len(pool))][: args.n]
    a, b, c = sel.T
    t = a + b + c
    ids = nl.encode(tok, [nl.three_prompt(*x) for x in sel], dev)

    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[:, -1].float(), -1)
    gold = num_ids[torch.tensor(t, device=dev)]
    gold_lp = lp.gather(1, gold[:, None])[:, 0]
    rank = (lp > gold_lp[:, None]).sum(1) + 1
    first = lp.argmax(-1)

    gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)

    def cat_first(i):
        tokstr = tok.convert_ids_to_tokens(int(first[i]))
        for name, val in (("total", t[i]), ("a+b", a[i] + b[i]), ("b+c", b[i] + c[i]), ("a+c", a[i] + c[i]),
                          ("a", a[i]), ("b", b[i]), ("c", c[i])):
            if int(first[i]) == int(num_ids[val]):
                return name
        if int(first[i]) in set(num_ids.tolist()):
            return "other_number"
        return f"non-number:{tokstr}"

    firsts = Counter(cat_first(i) for i in range(len(sel)))
    ints = []
    for i, s in enumerate(texts):
        m = re.search(r"-?\d+", s)
        v = int(m.group()) if m else None
        ints.append("total" if v == t[i] else ("a+b" if v == a[i] + b[i] else ("none" if v is None else "other")))
    first_int = Counter(ints)
    res = {
        "n": int(len(sel)),
        "first_token_acc": float((first == gold).float().mean()),
        "first_integer_in_generation_acc": first_int["total"] / len(sel),
        "gold_first_token_prob_mean": float(gold_lp.exp().mean()),
        "gold_rank_median": float(rank.float().median()),
        "gold_in_top5": float((rank <= 5).float().mean()),
        "first_token_categories": dict(firsts.most_common()),
        "first_integer_categories": dict(first_int.most_common()),
        "examples": [{"problem": f"{x[0]}+{x[1]}+{x[2]}", "total": int(x.sum()), "generation": texts[i]} for i, x in enumerate(sel[:25])],
    }
    nl.save_json(res, out / "exp2d.json")
    print({k: v for k, v in res.items() if k != "examples"})
    for e in res["examples"][:10]:
        print(e)


if __name__ == "__main__":
    main()
