"""Idea 1, follow-up: where is the tens digit?

exp1 on main: editing the helix subspace moves the answer by +-1/+-2 (65-79%) but never by +-10/+-20,
although swapping the whole operand state does. This script fits several number bases at the operand
slots of "Output ONLY a number.{a}+{b}=" and edits within each, block by block:

  helix       cos/sin(2 pi x/T), T in {2,5,10,100} (+ x/100)            [as exp1]
  digit_1hot  one-hot(tens) (+) one-hot(units)                         (20 cols)
  digit_circ  cos/sin(2 pi tens/10), cos/sin(2 pi units/10)            (4 cols)
  both        helix (+) digit_1hot

Edits: whole basis, tens block only, units block only; controls: same-norm random direction
orthogonal to the basis subspace, and the whole-state swap. Delta sets are chosen so that "units"
shifts never carry (units stays in 0..9) and "tens" shifts leave the units digit alone.

    python tools/exp1b_digits.py --model <dir> --out <dir> [--n 200] [--layers 0-20] [--step 2]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl

DELTAS = {"units": (-3, -2, -1, 1, 2, 3), "tens": (-30, -20, -10, 10, 20, 30)}


def basis(name: str, x) -> tuple:
    """Feature matrix and column blocks {'tens': [...], 'units': [...]} for integers x in 0..99."""
    x = np.asarray(x); t, u = x // 10, x % 10
    if name == "helix":
        F, names = nl.features(x)
        groups = nl.component_columns(names)
        # tens-like = slow components; units-like = the periods that divide 10
        return F, {"tens": groups["T100"] + groups["lin"], "units": groups["T2"] + groups["T5"] + groups["T10"]}
    if name == "digit_1hot":
        return np.concatenate([np.eye(10)[t], np.eye(10)[u]], 1), {"tens": list(range(10)), "units": list(range(10, 20))}
    if name == "digit_circ":
        F = np.stack([np.cos(2 * np.pi * t / 10), np.sin(2 * np.pi * t / 10),
                      np.cos(2 * np.pi * u / 10), np.sin(2 * np.pi * u / 10)], 1)
        return F, {"tens": [0, 1], "units": [2, 3]}
    if name == "both":
        Fh, bh = basis("helix", x); Fd, bd = basis("digit_1hot", x); k = Fh.shape[1]
        return np.concatenate([Fh, Fd], 1), {"tens": bh["tens"] + [k + i for i in bd["tens"]],
                                             "units": bh["units"] + [k + i for i in bd["units"]]}
    raise ValueError(name)


BASES = ("helix", "digit_1hot", "digit_circ", "both")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layers", default="0-20")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug_all", action="store_true", help="skip the answered-correctly filter (mechanics tests only)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    lo, hi = (int(v) for v in args.layers.split("-"))
    layers = list(range(lo, min(hi, len(model.model.layers) - 1) + 1, args.step))
    P = nl.positions(tok, "binary")
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    rng = np.random.default_rng(args.seed)

    pairs = np.array([(a, b) for a in range(100) for b in range(100) if a + b <= 99])
    ids_all = nl.encode(tok, [nl.binary_prompt(a, b) for a, b in pairs], dev)
    pred = torch.cat([nl.last_logprobs(model, ids_all[i:i + args.bs]).argmax(-1) for i in range(0, len(ids_all), args.bs)])
    correct = (pred == num_ids[torch.tensor(pairs.sum(1), device=dev)]).cpu().numpy()
    base_acc = float(correct.mean())
    if args.debug_all:
        correct = np.ones_like(correct)

    ids_a = nl.encode(tok, [nl.binary_prompt(a, 10) for a in range(100)], dev)
    HA = {L: v[:, 0] for L, v in nl.collect(model, ids_a, layers, [P["A"]], args.bs).items()}
    fit_idx = rng.choice(len(pairs), size=min(2000, len(pairs)), replace=False)
    fp = pairs[fit_idx]
    HB = {L: v[:, 0] for L, v in nl.collect(model, ids_all[fit_idx], layers, [P["B"]], args.bs).items()}

    fits, maps = [], {}
    for L in layers:
        for bname in BASES:
            Fa, blocks = basis(bname, np.arange(100))
            muA, WA = nl.fit_map(Fa, HA[L])
            FbA, _ = basis(bname, fp[:, 0]); FbB, _ = basis(bname, fp[:, 1])
            Fab = np.concatenate([FbA, FbB], 1)
            _, Wab = nl.fit_map(Fab, HB[L])
            maps[(L, bname)] = {"A": WA, "B": Wab[FbA.shape[1]:], "blocks": blocks}
            fits.append({"layer": L, "basis": bname, "r2_A": nl.cv_r2(Fa, HA[L]),
                         "r2_B_joint": nl.cv_r2(Fab, HB[L], groups=fp[:, 1])})
        print(f"fits L{L} ({time.time() - t0:.0f}s)")
    nl.save_json({"base_acc": base_acc, "fits": fits}, out / "fits.json")

    rows = []
    for slot in ("A", "B"):
        pos = P[slot]
        for kind, deltas in DELTAS.items():
            for delta in deltas:
                def valid(a, b):
                    x = a if slot == "A" else b
                    if not (0 <= x + delta <= 99 and a + b + delta <= 99):
                        return False
                    if kind == "units":
                        return 0 <= x % 10 + delta <= 9
                    return True
                ok = [(a, b) for (a, b), c in zip(pairs, correct) if c and valid(a, b)]
                if len(ok) < 20:
                    continue
                sel = np.array(ok)[rng.choice(len(ok), size=min(args.n, len(ok)), replace=False)]
                a, b = sel[:, 0], sel[:, 1]
                x = a if slot == "A" else b
                ids = nl.encode(tok, [nl.binary_prompt(i, j) for i, j in sel], dev)
                tgt = num_ids[torch.tensor(a + b + delta, device=dev)]
                orig = num_ids[torch.tensor(a + b, device=dev)]
                dp = np.stack([a + delta, b], 1) if slot == "A" else np.stack([a, b + delta], 1)
                donor = {L: v[:, 0] for L, v in nl.collect(model, nl.encode(tok, [nl.binary_prompt(i, j) for i, j in dp], dev), layers, [pos], args.bs).items()}
                own = {L: v[:, 0] for L, v in nl.collect(model, ids, layers, [pos], args.bs).items()}
                for L in layers:
                    conds = {"full_swap": donor[L] - own[L]}
                    for bname in BASES:
                        m = maps[(L, bname)]; W = m[slot]
                        dF = basis(bname, x + delta)[0] - basis(bname, x)[0]
                        full = dF @ W
                        conds[f"{bname}:all"] = full
                        for blk in ("tens", "units"):
                            cols = m["blocks"][blk]
                            conds[f"{bname}:{blk}"] = dF[:, cols] @ W[cols]
                        Q = nl.row_space_projector(W)
                        noise = rng.standard_normal(full.shape); noise -= (noise @ Q) @ Q.T
                        nz = np.linalg.norm(full, axis=1, keepdims=True)
                        conds[f"{bname}:random_orth"] = noise * nz / np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-9)
                    for cname, d in conds.items():
                        dt = torch.tensor(d, dtype=torch.float32, device=dev)
                        lps = []
                        for i in range(0, len(ids), args.bs):
                            with nl.patch(model, L, pos, delta=dt[i:i + args.bs]):
                                lps.append(nl.last_logprobs(model, ids[i:i + args.bs]))
                        am = torch.cat(lps).argmax(-1)
                        rows.append({"slot": slot, "kind": kind, "delta": delta, "layer": L, "cond": cname, "n": len(sel),
                                     "success": float((am == tgt).float().mean()), "stay": float((am == orig).float().mean()),
                                     "edit_norm": float(np.linalg.norm(d, axis=1).mean())})
                print(f"{slot} {kind} {delta:+d} ({time.time() - t0:.0f}s)")
    if not rows:
        (out / "summary.md").write_text(f"# exp1b\n\nbase acc {base_acc:.3f}: too few correct problems; fits only.\n")
        return
    with open(out / "steer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    lines = [f"# exp1b: where is the tens digit?\n\nbase acc {base_acc:.3f}\n\n## held-out R^2 (slot A), by layer\n\n| basis | "
             + " | ".join(f"L{L}" for L in layers) + " |\n|---|" + "---|" * len(layers) + "\n"]
    for bname in BASES:
        lines.append(f"| {bname} | " + " | ".join(f"{next(f['r2_A'] for f in fits if f['layer'] == L and f['basis'] == bname):.2f}" for L in layers) + " |\n")
    for slot in ("A", "B"):
        lines.append(f"\n## slot {slot}: success, best layer per condition (mean over the Deltas of each kind)\n\n| cond | units shifts | tens shifts |\n|---|---|---|\n")
        conds = sorted({r["cond"] for r in rows if r["slot"] == slot})
        for cond in conds:
            cells = []
            for kind in ("units", "tens"):
                by_l = {}
                for r in rows:
                    if r["slot"] == slot and r["cond"] == cond and r["kind"] == kind:
                        by_l.setdefault(r["layer"], []).append(r["success"])
                if by_l:
                    L = max(by_l, key=lambda k: np.mean(by_l[k]))
                    cells.append(f"{np.mean(by_l[L]):.2f} (L{L})")
                else:
                    cells.append("-")
            lines.append(f"| {cond} | {cells[0]} | {cells[1]} |\n")
    (out / "summary.md").write_text("".join(lines))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
