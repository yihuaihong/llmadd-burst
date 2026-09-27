"""Idea 1: is the helix the variable the model reads when it adds?

For every layer L and operand slot (A, B) of "Output ONLY a number.{a}+{b}=":
  1. fit the residual at that slot as mu + f(x) W (f = helix basis) and score it by held-out R^2
     against nulls (shuffled labels, wrong periods, linear only);
  2. edit the residual inside the helix subspace so it encodes x+Delta instead of x, and check whether
     the answer moves from a+b to a+b+Delta. Controls: same-norm random direction orthogonal to the
     helix subspace, the real state of x+Delta (full swap = upper bound), the real difference projected
     onto the helix subspace, and edits restricted to one period (T2/T5/T10/T100/linear).

Only problems the unedited model answers correctly are used.

    python tools/exp1_steer.py --model <dir> --out <dir> [--n 200] [--layers 0-31]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl

DELTAS = (-20, -10, -5, -2, -1, 1, 2, 5, 10, 20)


def parse_layers(spec: str, n_layers: int) -> list:
    if spec == "all":
        return list(range(n_layers))
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="problems per Delta")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug_all", action="store_true", help="skip the answered-correctly filter (mechanics tests only)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    layers = parse_layers(args.layers, len(model.model.layers))
    P = nl.positions(tok, "binary")
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    rng = np.random.default_rng(args.seed)

    # ---- base accuracy on every a+b<=99 problem
    pairs = np.array([(a, b) for a in range(100) for b in range(100) if a + b <= 99])
    ids_all = nl.encode(tok, [nl.binary_prompt(a, b) for a, b in pairs], dev)
    pred = []
    for i in range(0, len(ids_all), args.bs):
        pred.append(nl.last_logprobs(model, ids_all[i:i + args.bs]).argmax(-1))
    pred = torch.cat(pred).cpu().numpy()
    correct = pred == num_ids.cpu().numpy()[pairs.sum(1)]
    base_acc = float(correct.mean())
    print(f"base accuracy on {len(pairs)} problems a+b<=99: {base_acc:.3f}")
    if args.debug_all:
        correct = np.ones_like(correct)

    # ---- representations used for the fits
    # A slot: depends only on the prefix, so one prompt per a.
    ids_a = nl.encode(tok, [nl.binary_prompt(a, 10) for a in range(100)], dev)
    HA = {L: v[:, 0] for L, v in nl.collect(model, ids_a, layers, [P["A"]], args.bs).items()}
    # B slot: depends on a and b; fit jointly on a random sample of problems.
    fit_idx = rng.choice(len(pairs), size=min(2000, len(pairs)), replace=False)
    fit_pairs = pairs[fit_idx]
    HB = {L: v[:, 0] for L, v in nl.collect(model, ids_all[fit_idx], layers, [P["B"]], args.bs).items()}

    Fa, names = nl.features(np.arange(100))
    comps = nl.component_columns(names)
    Fwrong, _ = nl.features(np.arange(100), periods=(3, 7, 11, 13))
    Flin, _ = nl.features(np.arange(100), periods=())
    FbA, _ = nl.features(fit_pairs[:, 0]); FbB, _ = nl.features(fit_pairs[:, 1])

    fit_rows, maps = [], {}
    for L in layers:
        perm_r2 = np.mean([nl.cv_r2(Fa[rng.permutation(100)], HA[L], seed=s) for s in range(3)])
        row = {"layer": L, "slot": "A", "r2_helix": nl.cv_r2(Fa, HA[L]), "r2_wrong_periods": nl.cv_r2(Fwrong, HA[L]),
               "r2_linear_only": nl.cv_r2(Flin, HA[L]), "r2_shuffled": float(perm_r2)}
        fit_rows.append(row)
        muA, WA = nl.fit_map(Fa, HA[L])
        # B: H = mu + f(a) Wa + f(b) Wb ; keep Wb. Held-out R^2 of the b block over and above a.
        Fab = np.concatenate([FbA, FbB], 1)
        muB, Wab = nl.fit_map(Fab, HB[L])
        WB = Wab[FbA.shape[1]:]
        fit_rows.append({"layer": L, "slot": "B", "r2_helix": nl.cv_r2(Fab, HB[L], groups=fit_pairs[:, 1]),
                         "r2_a_only": nl.cv_r2(FbA, HB[L], groups=fit_pairs[:, 1])})
        maps[L] = {"A": WA, "B": WB}
        print(f"L{L:2d} R2 A helix={row['r2_helix']:.3f} wrong={row['r2_wrong_periods']:.3f} "
              f"lin={row['r2_linear_only']:.3f} shuf={row['r2_shuffled']:.3f}")
    nl.save_json({"base_acc": base_acc, "fits": fit_rows}, out / "fits.json")

    # ---- steering
    rows = []
    for slot in ("A", "B"):
        pos = P[slot]
        for delta in DELTAS:
            if slot == "A":
                ok = [(a, b) for (a, b), c in zip(pairs, correct) if c and 0 <= a + delta <= 99 and 0 <= a + b + delta <= 99]
            else:
                ok = [(a, b) for (a, b), c in zip(pairs, correct) if c and 0 <= b + delta <= 99 and 0 <= a + b + delta <= 99]
            if len(ok) < 20:
                continue
            sel = np.array(ok)[rng.choice(len(ok), size=min(args.n, len(ok)), replace=False)]
            a, b = sel[:, 0], sel[:, 1]
            x = a if slot == "A" else b
            ids = nl.encode(tok, [nl.binary_prompt(i, j) for i, j in sel], dev)
            tgt = num_ids[torch.tensor(a + b + delta, device=dev)]
            orig = num_ids[torch.tensor(a + b, device=dev)]
            dF = nl.features(x + delta)[0] - nl.features(x)[0]           # [n, k]
            # real states of the edited operand at this slot (donor prompts), all layers at once
            donor_pairs = np.stack([a + delta, b], 1) if slot == "A" else np.stack([a, b + delta], 1)
            donor_ids = nl.encode(tok, [nl.binary_prompt(i, j) for i, j in donor_pairs], dev)
            donor = {L: v[:, 0] for L, v in nl.collect(model, donor_ids, layers, [pos], args.bs).items()}
            own = {L: v[:, 0] for L, v in nl.collect(model, ids, layers, [pos], args.bs).items()}
            for L in layers:
                W = maps[L][slot]
                Q = nl.row_space_projector(W)                          # [d, r]
                helix = dF @ W
                conds = {"helix": helix}
                for cname, cols in comps.items():
                    conds[f"only_{cname}"] = dF[:, cols] @ W[cols]
                noise = rng.standard_normal(helix.shape)
                noise -= (noise @ Q) @ Q.T
                noise *= np.linalg.norm(helix, axis=1, keepdims=True) / np.linalg.norm(noise, axis=1, keepdims=True)
                conds["random_orth"] = noise
                real = donor[L] - own[L]
                conds["real_diff_in_helix_subspace"] = (real @ Q) @ Q.T
                conds["full_swap"] = real
                for cname, d in conds.items():
                    dt = torch.tensor(d, dtype=torch.float32, device=dev)
                    lps = []
                    for i in range(0, len(ids), args.bs):
                        with nl.patch(model, L, pos, delta=dt[i:i + args.bs]):
                            lps.append(nl.last_logprobs(model, ids[i:i + args.bs]))
                    lp = torch.cat(lps)
                    am = lp.argmax(-1)
                    rows.append({
                        "slot": slot, "layer": L, "delta": delta, "cond": cname, "n": len(sel),
                        "success": float((am == tgt).float().mean()), "stay": float((am == orig).float().mean()),
                        "lp_target": float(lp.gather(1, tgt[:, None]).mean()),
                        "lp_orig": float(lp.gather(1, orig[:, None]).mean()),
                        "edit_norm": float(np.linalg.norm(d, axis=1).mean()),
                        "state_norm": float(np.linalg.norm(own[L], axis=1).mean()),
                    })
            print(f"slot {slot} delta {delta:+d}: done ({time.time() - t0:.0f}s)")
    if not rows:
        nl.save_json({"base_acc": base_acc, "fits": fit_rows, "note": "too few correct problems to steer"}, out / "fits.json")
        (out / "summary.md").write_text(f"# exp1\n\nbase accuracy {base_acc:.3f}: too few correct problems to steer; fits only.\n")
        return
    with open(out / "steer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summarize(rows, fit_rows, base_acc, out)
    print(f"total {time.time() - t0:.0f}s")


def summarize(rows: list, fit_rows: list, base_acc: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [f"# exp1 phase-rotation steering\n", f"base accuracy (a+b<=99): {base_acc:.3f}\n"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fa = [r for r in fit_rows if r["slot"] == "A"]
    for key in ("r2_helix", "r2_wrong_periods", "r2_linear_only", "r2_shuffled"):
        axes[0].plot([r["layer"] for r in fa], [r[key] for r in fa], label=key)
    axes[0].set_title("slot A: held-out R^2 of helix map"); axes[0].set_xlabel("layer"); axes[0].legend()
    for ax, slot in zip(axes[1:], ("A", "B")):
        for cond in ("helix", "real_diff_in_helix_subspace", "full_swap", "random_orth"):
            by_layer = {}
            for r in rows:
                if r["slot"] == slot and r["cond"] == cond:
                    by_layer.setdefault(r["layer"], []).append(r["success"])
            ls = sorted(by_layer)
            ax.plot(ls, [np.mean(by_layer[L]) for L in ls], label=cond)
        ax.set_title(f"slot {slot}: answer moves to a+b+Delta (mean over Delta)"); ax.set_xlabel("layer")
        ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(out / "exp1_overview.png", dpi=120); plt.close(fig)

    for slot in ("A", "B"):
        sub = [r for r in rows if r["slot"] == slot and r["cond"] == "helix"]
        if not sub:
            continue
        best = max({r["layer"] for r in sub}, key=lambda L: np.mean([r["success"] for r in sub if r["layer"] == L]))
        lines.append(f"\n## slot {slot}: best helix layer {best}\n")
        lines.append("| cond | " + " | ".join(f"Δ{d:+d}" for d in DELTAS) + " |\n")
        lines.append("|---|" + "---|" * len(DELTAS) + "\n")
        conds = sorted({r["cond"] for r in rows if r["slot"] == slot})
        for cond in conds:
            vals = []
            for d in DELTAS:
                m = [r["success"] for r in rows if r["slot"] == slot and r["cond"] == cond and r["layer"] == best and r["delta"] == d]
                vals.append(f"{m[0]:.2f}" if m else "-")
            lines.append(f"| {cond} | " + " | ".join(vals) + " |\n")
    (out / "summary.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
