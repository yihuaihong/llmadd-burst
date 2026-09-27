"""Idea 2, follow-up: is the late running sum at the second '+' the ordinary "answer" computation?

Hypothesis: in "a+b+c=" the second '+' closes "a+b" the way '=' closes "a+b=", so the same circuit
writes a+b there (from ~layer 21, like the answer at '=') and the result is then ignored.

Test per layer: fit an a+b readout (kernel ridge onto the helix features) on '=' states of the binary
prompts "a+b=", apply it unchanged to the second-'+' states of "a+b+c=" (and the reverse), on (a,b)
pairs disjoint from the training pairs. Compare with the within-format readouts (the ceiling), and
report the cosine similarity of the two states for the same (a,b) (mean-centred per format).

    python tools/exp2c_transfer.py --model <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl


def fit(X: torch.Tensor, Y: torch.Tensor, alphas=nl.ALPHAS):
    """Kernel ridge X -> Y with alpha by closed-form leave-one-out; returns a predict function."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    lam, U = torch.linalg.eigh(Xc @ Xc.T)
    lam = lam.clamp_min(0)
    UY = U.T @ Yc
    best, best_err = None, None
    for a in alphas:
        h = lam / (lam + a)
        err = (((Yc - U @ (h[:, None] * UY)) / (1 - (U ** 2) @ h).clamp_min(1e-6)[:, None]) ** 2).mean()
        if best_err is None or err < best_err:
            best, best_err = a, err
    B = Xc.T @ (U @ (UY / (lam + best)[:, None]))
    return lambda Z: (Z - xm) @ B + ym


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=1600)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    layers = list(range(len(model.model.layers)))
    Pb, Pt = nl.positions(tok, "binary"), nl.positions(tok, "three")
    rng = np.random.default_rng(args.seed)

    allpairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100) if a + b <= 89])
    pairs = allpairs[rng.permutation(len(allpairs))][: args.pairs]
    c = np.array([rng.integers(10, 100 - a - b) for a, b in pairs])
    s = pairs.sum(1)
    Hb = nl.collect(model, nl.encode(tok, [nl.binary_prompt(a, b) for a, b in pairs], dev), layers, [Pb["eq"]], args.bs)
    Ht = nl.collect(model, nl.encode(tok, [nl.three_prompt(a, b, ci) for (a, b), ci in zip(pairs, c)], dev), layers, [Pt["op2"]], args.bs)
    print(f"collected ({time.time() - t0:.0f}s)")

    half = len(pairs) // 2
    tr, te = np.arange(half), np.arange(half, len(pairs))            # disjoint (a, b) pairs
    cand = np.arange(0, 100)
    Fc = torch.tensor(nl.features(cand)[0], dtype=torch.float64, device=dev)
    Fs = torch.tensor(nl.features(s)[0], dtype=torch.float64, device=dev)
    s_t = torch.tensor(s, device=dev)

    def acc(pred, idx):
        dec = torch.as_tensor(cand, device=dev)[torch.cdist(pred, Fc).argmin(1)]
        return float((dec == s_t[idx]).float().mean())

    rows = []
    for L in layers:
        B = torch.tensor(Hb[L][:, 0], dtype=torch.float64, device=dev)
        T = torch.tensor(Ht[L][:, 0], dtype=torch.float64, device=dev)
        ti, ei = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
        fb, ft = fit(B[ti], Fs[ti]), fit(T[ti], Fs[ti])
        Bc, Tc = B - B[ti].mean(0), T - T[ti].mean(0)
        cos = torch.nn.functional.cosine_similarity(Bc[ei], Tc[ei], dim=1).mean()
        rows.append({"layer": L,
                     "bin_eq->bin_eq": acc(fb(B[ei]), ei), "three_op2->three_op2": acc(ft(T[ei]), ei),
                     "bin_eq->three_op2": acc(fb(T[ei]), ei), "three_op2->bin_eq": acc(ft(B[ei]), ei),
                     # same, with each format re-centred on its own training mean (tests directions, not offsets)
                     "bin_eq->three_op2_centred": acc(fb(T[ei] - T[ti].mean(0) + B[ti].mean(0)), ei),
                     "three_op2->bin_eq_centred": acc(ft(B[ei] - B[ti].mean(0) + T[ti].mean(0)), ei),
                     "cos_same_pair": float(cos)})
    nl.save_json({"pairs": int(len(pairs)), "rows": rows}, out / "exp2c.json")
    lines = ["# exp2c: does an a+b readout trained at '=' of a+b= read the second '+' of a+b+c=?\n\n",
             "| layer | bin '=' -> bin '=' | three op2 -> three op2 | bin '=' -> three op2 | three op2 -> bin '=' | centred: bin->three | centred: three->bin | cos(same pair) |\n",
             "|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        lines.append(f"| {r['layer']} | {r['bin_eq->bin_eq']:.2f} | {r['three_op2->three_op2']:.2f} | "
                     f"{r['bin_eq->three_op2']:.2f} | {r['three_op2->bin_eq']:.2f} | {r['bin_eq->three_op2_centred']:.2f} | "
                     f"{r['three_op2->bin_eq_centred']:.2f} | {r['cos_same_pair']:.2f} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
