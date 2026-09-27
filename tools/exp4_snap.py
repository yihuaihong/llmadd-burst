"""Does putting operand representations ON the ideal manifold improve arithmetic? (inference-time test)

Training-free version of the manifold hypothesis. For each operand slot (A, B, and C in three-term
problems) and each layer L, fit a geometric model of the slot's number representation on the
checkpoint's own in-context states (per-number means over contexts):
    h ~ mu + F_G(x) W_G,     G in {helix, digit}
with S_G = span(rows of W_G) the manifold subspace and m_G(x) = mu + F_G(x) W_G the ideal point.
Then, on every problem (not only the ones the model gets right), edit the operand states at layer L:
    snap_G       h' = h + P_S (m_G(x) - h)          manifold coordinates set to the ideal point for x
    classmean    h' = h + P_S (hbar(x) - h)         same subspace, the model's own mean state for x
    shuffled_G   h' = h + P_S (m_G(pi(x)) - h)      same subspace, ideal point of a wrong number (must hurt)
    random       same-norm random edit orthogonal to S_G (as big as snap_G)
and compare accuracy with the unedited model. Natural format "Q: a + b = " / "Q: a + b + c = ".

    python tools/exp4_snap.py --model <dir> --out <dir> [--layers 0-20] [--step 2]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl


def two_prompt(a, b): return f"Q: {a} + {b} = "
def three_prompt(a, b, c): return f"Q: {a} + {b} + {c} = "


def geom_features(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if name == "helix":
        F = nl.features(x)[0]
    elif name == "digit":
        F = np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1)
    else:
        raise ValueError(name)
    return F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-20")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    lo, hi = (int(v) for v in args.layers.split("-"))
    layers = list(range(lo, min(hi, len(model.model.layers) - 1) + 1, args.step))
    rng = np.random.default_rng(args.seed)
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    t2 = tok.convert_ids_to_tokens(tok(two_prompt(23, 45))["input_ids"])
    t3 = tok.convert_ids_to_tokens(tok(three_prompt(23, 45, 12))["input_ids"])
    assert t2[-6:] == ["23", "Ġ+", "Ġ", "45", "Ġ=", "Ġ"], t2
    assert t3[-9:] == ["23", "Ġ+", "Ġ", "45", "Ġ+", "Ġ", "12", "Ġ=", "Ġ"], t3
    pos2 = {"A": len(t2) - 6, "B": len(t2) - 3}
    pos3 = {"A": len(t3) - 9, "B": len(t3) - 6, "C": len(t3) - 3}
    perm = np.arange(100); perm[10:] = 10 + rng.permutation(90)       # wrong-number map on 10..99

    # problems: all two-digit operands; answers stay single tokens
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    two = pairs[rng.permutation(len(pairs))][: args.n]
    trip = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 199])
    three = trip[rng.permutation(len(trip))][: args.n]
    tasks = {
        "two": (nl.encode(tok, [two_prompt(*x) for x in two], dev), two, pos2),
        "three": (nl.encode(tok, [three_prompt(*x) for x in three], dev), three, pos3),
    }

    # manifold fits per (task, slot, layer) from the checkpoint's own states on a separate fit set
    fit_two = pairs[rng.permutation(len(pairs))][:3000]
    fit_three = trip[rng.permutation(len(trip))][:3000]
    fits = {}
    for tname, fitset, pos, pf in (("two", fit_two, pos2, two_prompt), ("three", fit_three, pos3, three_prompt)):
        ids = nl.encode(tok, [pf(*x) for x in fitset], dev)
        slots = list(pos)
        H = nl.collect(model, ids, layers, [pos[s] for s in slots], args.bs)
        for j, s in enumerate(slots):
            x = fitset[:, j]
            for L in layers:
                hs = H[L][:, j]
                assert all((x == v).any() for v in range(10, 100)), f"fit set misses a value in slot {s}"
                hbar = np.stack([hs[x == v].mean(0) for v in range(10, 100)])       # per-number means
                mu = hbar.mean(0)
                entry = {"hbar": hbar, "mu": mu}
                for G in ("helix", "digit"):
                    F = geom_features(G, np.arange(10, 100))
                    m0, W = nl.fit_map(F, hbar)
                    entry[G] = {"W": W, "m0": m0, "Q": nl.row_space_projector(W),
                                "r2": nl.cv_r2(F, hbar)}
                fits[(tname, s, L)] = entry
    print(f"fits done ({time.time() - t0:.0f}s)")

    def accuracy(ids, gold, edits=None):
        """edits: list of (layer, pos, delta[n, d]) applied together."""
        am = []
        for i in range(0, len(ids), args.bs):
            handles = []
            for (L, p, d) in (edits or []):
                dd = d[i:i + args.bs]

                def hook(_m, _i, o, p=p, dd=dd):
                    h = nl._hidden(o).clone(); h[:, p, :] = h[:, p, :] + dd.to(h.dtype); return nl._with_hidden(o, h)
                handles.append(model.model.layers[L].register_forward_hook(hook))
            try:
                am.append(nl.last_logprobs(model, ids[i:i + args.bs]).argmax(-1))
            finally:
                for h in handles:
                    h.remove()
        return float((torch.cat(am) == gold).float().mean())

    rows = []
    for tname, (ids, probs, pos) in tasks.items():
        gold = num_ids[torch.tensor(probs.sum(1), device=dev)]
        base = accuracy(ids, gold)
        rows.append({"task": tname, "layer": -1, "cond": "base", "acc": base})
        own = nl.collect(model, ids, layers, [pos[s] for s in pos], args.bs)
        for L in layers:
            for G in ("helix", "digit"):
                conds = {f"snap_{G}": [], f"shuffled_{G}": [], f"random_{G}": [], f"classmean_{G}": []}
                for j, s in enumerate(pos):
                    e = fits[(tname, s, L)]; g = e[G]; Q = g["Q"]
                    x = probs[:, j]
                    h = own[L][:, j]
                    ideal = g["m0"] + geom_features(G, x) @ g["W"]
                    wrong = g["m0"] + geom_features(G, perm[x]) @ g["W"]
                    cm = e["hbar"][x - 10]
                    d_snap = ((ideal - h) @ Q) @ Q.T
                    noise = rng.standard_normal(h.shape); noise -= (noise @ Q) @ Q.T
                    noise *= np.linalg.norm(d_snap, axis=1, keepdims=True) / np.linalg.norm(noise, axis=1, keepdims=True)
                    for cname, d in ((f"snap_{G}", d_snap), (f"shuffled_{G}", ((wrong - h) @ Q) @ Q.T),
                                     (f"random_{G}", noise), (f"classmean_{G}", ((cm - h) @ Q) @ Q.T)):
                        conds[cname].append((L, pos[s], torch.tensor(d, dtype=torch.float32, device=dev)))
                for cname, edits in conds.items():
                    rows.append({"task": tname, "layer": L, "cond": cname, "acc": accuracy(ids, gold, edits),
                                 "r2_mean_slots": float(np.mean([fits[(tname, s, L)][G]["r2"] for s in pos])),
                                 "edit_norm": float(np.mean([e[2].norm(dim=1).mean().item() for e in edits])),
                                 "state_norm": float(np.mean([np.linalg.norm(own[L][:, j], axis=1).mean() for j in range(len(pos))]))})
            print(f"{tname} L{L} ({time.time() - t0:.0f}s)")
    nl.save_json(rows, out / "exp4.json")

    lines = ["# exp4: snap operand representations onto the ideal manifold (inference time)\n"]
    for tname in tasks:
        base = next(r["acc"] for r in rows if r["task"] == tname and r["cond"] == "base")
        lines.append(f"\n## {tname}-term: base accuracy {base:.3f}\n\n| cond | " + " | ".join(f"L{L}" for L in layers) + " |\n|---|" + "---|" * len(layers) + "\n")
        for cname in ("snap_helix", "classmean_helix", "shuffled_helix", "random_helix", "snap_digit", "classmean_digit", "shuffled_digit", "random_digit"):
            vals = [next((r["acc"] for r in rows if r["task"] == tname and r["cond"] == cname and r["layer"] == L), np.nan) for L in layers]
            lines.append(f"| {cname} | " + " | ".join(f"{v - base:+.3f}" for v in vals) + " |\n")
        lines.append("\n(entries: accuracy minus base)\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
