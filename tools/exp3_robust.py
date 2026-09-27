"""Robustness + mechanism for the three-term "right inside, wrong tens outside" finding (F6).

1. Prompt formats for a+b+c: the terse training format, the spaced "Q: a + b + c = " format, and a
   2-shot format. Per format: model first-token accuracy, error histogram, error rate by the tens
   digit of the total, and a probe on the last-position state (layers 20..31, folds grouped by the
   (a,b) pair) -> P(probe right | model wrong).
2. Two-term a+b (terse and spaced): accuracy by the tens digit of the sum - is the 5x/6x dip there too?
3. Unembedding geometry for the number tokens 0..99: cosine between the effective readout vectors
   (lm_head rows scaled by the final-norm weight) of n and n-10, by the tens digit of n.

    python tools/exp3_robust.py --model <dir> --out <dir>
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import numlib as nl

FORMATS = {
    "terse": lambda a, b, c: f"{nl.PREFIX}{a}+{b}+{c}=",
    "spaced": lambda a, b, c: f"Q: {a} + {b} + {c} = ",
    "two_shot": lambda a, b, c: f"12+31+20=63\n25+14+33=72\n{a}+{b}+{c}=",
}
BIN_FORMATS = {"terse": lambda a, b: f"{nl.PREFIX}{a}+{b}=", "spaced": lambda a, b: f"Q: {a} + {b} = "}


def by_tens(values: np.ndarray, wrong: np.ndarray) -> dict:
    out = {}
    for d in sorted(set((values // 10).tolist())):
        m = values // 10 == d
        out[str(d)] = {"n": int(m.sum()), "wrong": float(wrong[m].mean())}
    return out


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
    n_layers = len(model.model.layers)
    rng = np.random.default_rng(args.seed)
    num = nl.number_token_ids(tok, 999)
    inv = {int(v): k for k, v in enumerate(num)}
    res = {"three_term": {}, "two_term": {}}

    pool = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 99])
    sel = pool[rng.permutation(len(pool))][: args.n]
    t = sel.sum(1)
    pair = sel[:, 0] * 100 + sel[:, 1]
    layers = list(range(20, n_layers))
    for name, fmt in FORMATS.items():
        ids = nl.encode(tok, [fmt(*x) for x in sel], dev)
        first = torch.cat([nl.last_logprobs(model, ids[i:i + args.bs]).argmax(-1) for i in range(0, len(ids), args.bs)]).cpu().numpy()
        said = np.array([inv.get(int(x), -1) for x in first])
        wrong = said != t
        H = nl.collect(model, ids, layers, [ids.shape[1] - 1], args.bs)
        probe = {}
        for L in layers:
            r = nl.probe_accuracy(H[L][:, 0], t, np.arange(100), groups=pair, return_pred=True)
            probe[L] = r["pred"] == t
        Lb = max(probe, key=lambda L: probe[L].mean())
        err = said[wrong & (said >= 0)] - t[wrong & (said >= 0)]
        res["three_term"][name] = {
            "prompt_example": fmt(40, 18, 25), "acc": float(1 - wrong.mean()), "says_number": float((said >= 0).mean()),
            "error_hist_top": {int(k): v for k, v in Counter(err.tolist()).most_common(8)},
            "wrong_by_tens_of_total": by_tens(t, wrong),
            "probe_best_layer": int(Lb), "probe_acc": float(probe[Lb].mean()),
            "P(probe right | model wrong)": float(probe[Lb][wrong].mean()) if wrong.any() else float("nan"),
        }
        print(name, {k: v for k, v in res["three_term"][name].items() if k != "wrong_by_tens_of_total"})

    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100) if a + b <= 99])
    s = pairs.sum(1)
    for name, fmt in BIN_FORMATS.items():
        ids = nl.encode(tok, [fmt(a, b) for a, b in pairs], dev)
        first = torch.cat([nl.last_logprobs(model, ids[i:i + args.bs]).argmax(-1) for i in range(0, len(ids), args.bs)]).cpu().numpy()
        said = np.array([inv.get(int(x), -1) for x in first])
        wrong = said != s
        res["two_term"][name] = {"acc": float(1 - wrong.mean()), "wrong_by_tens_of_sum": by_tens(s, wrong)}
        print(name, "two-term acc", res["two_term"][name]["acc"])

    with torch.no_grad():
        W = model.lm_head.weight[torch.tensor(num[:100], device=dev)].float() * model.model.norm.weight.float()
        Wn = torch.nn.functional.normalize(W, dim=1)
        cos = (Wn[10:] * Wn[:-10]).sum(1).cpu().numpy()          # cos(n, n-10) for n = 10..99
        cos1 = (Wn[1:] * Wn[:-1]).sum(1).cpu().numpy()            # cos(n, n-1)
    res["unembedding"] = {
        "cos_n_vs_n-10_by_tens": {str(d): float(cos[(np.arange(10, 100) // 10) == d].mean()) for d in range(1, 10)},
        "cos_n_vs_n-1_mean": float(cos1.mean()),
        "norm_by_tens": {str(d): float(W[d * 10:(d + 1) * 10].norm(dim=1).mean()) for d in range(10)},
    }
    nl.save_json(res, out / "exp3.json")

    lines = ["# exp3: robustness of the three-term tens readout error\n\n| format | acc | P(probe right \\| model wrong) | probe acc (L) | wrong by tens of total (3x..9x) |\n|---|---|---|---|---|\n"]
    for name, r in res["three_term"].items():
        bt = " ".join(f"{k}x:{v['wrong']:.2f}" for k, v in r["wrong_by_tens_of_total"].items())
        lines.append(f"| {name} | {r['acc']:.2f} | {r['P(probe right | model wrong)']:.2f} | {r['probe_acc']:.2f} (L{r['probe_best_layer']}) | {bt} |\n")
    lines.append("\n## two-term: wrong by tens of the sum\n\n")
    for name, r in res["two_term"].items():
        lines.append(f"- {name} (acc {r['acc']:.3f}): " + " ".join(f"{k}x:{v['wrong']:.2f}" for k, v in r["wrong_by_tens_of_sum"].items()) + "\n")
    lines.append("\n## unembedding: cos(readout(n), readout(n-10)) by tens of n\n\n" +
                 " ".join(f"{k}x:{v:.2f}" for k, v in res["unembedding"]["cos_n_vs_n-10_by_tens"].items()) +
                 f"\n\ncos(n, n-1) mean {res['unembedding']['cos_n_vs_n-1_mean']:.2f}\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
