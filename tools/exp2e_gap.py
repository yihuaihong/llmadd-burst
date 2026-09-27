"""Idea 2, follow-up: per problem, does the '=' state hold the right total when the model says a wrong one?

For each problem a+b+c= (all problems, not only correct ones): the model's first-token answer, and the
answer a probe decodes from the '=' state (kernel ridge onto helix features, folds grouped by (a,b),
so every probed pair is unseen). Per layer 18..31: probe accuracy, model accuracy, the joint table,
and units / tens digit accuracy for both. Plus the model's error structure (answer - total).

    python tools/exp2e_gap.py --model <dir> --out <dir> [--n 1500]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import numlib as nl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    P = nl.positions(tok, "three")
    rng = np.random.default_rng(args.seed)
    pool = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 99])
    sel = pool[rng.permutation(len(pool))][: args.n]
    t = sel.sum(1)
    ids = nl.encode(tok, [nl.three_prompt(*x) for x in sel], dev)
    num_ids = nl.number_token_ids(tok, 999)
    inv = {int(v): k for k, v in enumerate(num_ids)}
    first = torch.cat([nl.last_logprobs(model, ids[i:i + args.bs]).argmax(-1) for i in range(0, len(ids), args.bs)]).cpu().numpy()
    said = np.array([inv.get(int(x), -1) for x in first])          # -1: not a number token
    model_ok = said == t
    layers = list(range(18, len(model.model.layers)))
    H = nl.collect(model, ids, layers, [P["eq"]], args.bs)
    pair = sel[:, 0] * 100 + sel[:, 1]
    rows = []
    for L in layers:
        r = nl.probe_accuracy(H[L][:, 0], t, np.arange(100), groups=pair, return_pred=True)
        pr = r["pred"]; probe_ok = pr == t
        rows.append({
            "layer": L, "probe_acc": float(probe_ok.mean()), "model_acc": float(model_ok.mean()),
            "probe_ok_model_wrong": float((probe_ok & ~model_ok).mean()),
            "model_ok_probe_wrong": float((model_ok & ~probe_ok).mean()),
            "P(probe ok | model wrong)": float(probe_ok[~model_ok].mean()) if (~model_ok).any() else float("nan"),
            "probe_units_acc": float((pr % 10 == t % 10).mean()), "probe_tens_acc": float((pr // 10 == t // 10).mean()),
        })
    num = said >= 0
    err = said[num & ~model_ok] - t[num & ~model_ok]
    res = {
        "n": int(len(sel)), "model_acc": float(model_ok.mean()), "model_says_a_number": float(num.mean()),
        "model_units_acc": float((said[num] % 10 == t[num] % 10).mean()),
        "model_tens_acc": float((said[num] // 10 == t[num] // 10).mean()),
        "model_error_hist_top": {int(k): v for k, v in Counter(err.tolist()).most_common(12)},
        "wrong_with_units_right": float((said[num & ~model_ok] % 10 == t[num & ~model_ok] % 10).mean()) if (num & ~model_ok).any() else float("nan"),
        "layers": rows,
    }
    nl.save_json(res, out / "exp2e.json")
    best = max(rows, key=lambda r: r["probe_acc"])
    print({k: v for k, v in res.items() if k != "layers"})
    print("best probe layer:", best)


if __name__ == "__main__":
    main()
