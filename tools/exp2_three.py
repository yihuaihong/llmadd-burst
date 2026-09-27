"""Idea 2: in "Output ONLY a number.{a}+{b}+{c}=", is a+b already computed before c is read?

Slots: A, op1, B, op2 (the second '+'), C, eq. Three measurements per layer:
  (a) probes: cross-validated exact decoding of a, b, s=a+b, c, t=a+b+c from the residual at each slot
      (ridge onto the helix features, nearest candidate), with a shuffled-label null and a reference
      "how well can s be read LINEARLY from the true helix features of a and b alone";
  (b) interchange at slot B / op2 / A: swap in the residual of a donor prompt whose prefix is (a',b')
        - different sum  s' != s  -> does the answer become s'+c ?
        - same sum, different split a'+b' = s  -> does the answer stay s+c ?
      Both together separate "this slot carries the sum" from "this slot carries a and b";
  (c) helix steering of s at B / op2 with the s-block of a joint map mu + f(a)Wa + f(b)Wb + f(s)Ws.

Only problems the unedited model answers correctly are used.

    python tools/exp2_three.py --model <dir> --out <dir> [--n 300]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl

STEER_DELTAS = (-10, -5, -2, -1, 1, 2, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="problems for interchange/steering")
    ap.add_argument("--n_probe", type=int, default=1500, help="problems for probes and joint maps")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug_all", action="store_true", help="skip the answered-correctly filter (mechanics tests only)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    layers = list(range(len(model.model.layers)))
    P = nl.positions(tok, "three")
    slots = ["A", "op1", "B", "op2", "C", "eq"]
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    rng = np.random.default_rng(args.seed)

    # ---- problem pool: two-digit operands, all partial sums <= 99
    pool = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100)
                     if a + b + c <= 99])
    pool = pool[rng.permutation(len(pool))][:6000]
    ids = nl.encode(tok, [nl.three_prompt(*t) for t in pool], dev)
    pred = torch.cat([nl.last_logprobs(model, ids[i:i + args.bs]).argmax(-1) for i in range(0, len(ids), args.bs)])
    correct = (pred == num_ids[torch.tensor(pool.sum(1), device=dev)]).cpu().numpy()
    base_acc = float(correct.mean())
    print(f"base accuracy on {len(pool)} three-term problems: {base_acc:.3f}")
    if args.debug_all:
        correct = np.ones_like(correct)
    good = pool[correct]

    # ---- (a) probes: the same first n_probe problems of the pool for every checkpoint (not filtered on
    # correctness, so that early checkpoints that cannot add are measured on identical inputs)
    npb = min(args.n_probe, len(pool))
    H = nl.collect(model, ids[:npb], layers, [P[s] for s in slots], args.bs)   # {L: [n, slots, d]}
    g = pool[:npb]
    targets = {"a": g[:, 0], "b": g[:, 1], "s": g[:, 0] + g[:, 1], "c": g[:, 2], "t": g.sum(1)}
    cand = np.arange(0, 100)
    probe_rows = []
    # reference: s read linearly from the TRUE helix features of a and b (no model involved)
    Fab = np.concatenate([nl.features(g[:, 0])[0], nl.features(g[:, 1])[0]], 1)
    ref = nl.probe_accuracy(Fab, targets["s"], cand, alphas=(1e-3, 1e-1, 1e1))
    for L in layers:
        for j, slot in enumerate(slots):
            X = H[L][:, j]
            for tname, y in targets.items():
                r = nl.probe_accuracy(X, y, cand)
                probe_rows.append({"layer": L, "slot": slot, "target": tname, "acc": r["acc"]})
            if slot in ("op2", "B", "eq"):
                r = nl.probe_accuracy(X, rng.permutation(targets["s"]), cand)
                probe_rows.append({"layer": L, "slot": slot, "target": "s_shuffled", "acc": r["acc"]})
        print(f"probes L{L} done ({time.time() - t0:.0f}s)")
    with open(out / "probes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(probe_rows[0].keys())); w.writeheader(); w.writerows(probe_rows)

    # ---- (b) interchange and (c) steering, on problems the model answers correctly
    if len(good) < 50:
        nl.save_json({"base_acc": base_acc, "n_correct": int(len(good)), "n_probe": int(npb),
                      "ref_s_from_true_ab_features_linear": ref["acc"], "note": "too few correct problems for (b),(c)",
                      "interchange": [], "steer": []}, out / "exp2.json")
        plot(probe_rows, [], [], ref["acc"], base_acc, out)
        return
    n = min(args.n, len(good))
    sel = good[rng.choice(len(good), size=n, replace=False)]
    a, b, c = sel[:, 0], sel[:, 1], sel[:, 2]
    s = a + b
    ids_sel = nl.encode(tok, [nl.three_prompt(*t) for t in sel], dev)
    orig = num_ids[torch.tensor(s + c, device=dev)]

    def donor_split(same_sum: bool):
        """(a', b') per problem: same sum with a different split, or a different sum keeping s'+c<=99."""
        out_ab = []
        for ai, bi, ci in sel:
            si = ai + bi
            if same_sum:
                opts = [(x, si - x) for x in range(10, 100) if 10 <= si - x <= 99 and x != ai]
            else:
                opts = [(x, y) for x in range(10, 100) for y in range(10, 100)
                        if x + y != si and x + y + ci <= 99]
            out_ab.append(opts[rng.integers(len(opts))] if opts else (ai, bi))
        return np.array(out_ab)

    ic_rows = []
    own = nl.collect(model, ids_sel, layers, [P["A"], P["B"], P["op2"]], args.bs)
    for kind in ("diff_sum", "same_sum"):
        d_ab = donor_split(kind == "same_sum")
        valid = (d_ab[:, 0] != a) | (d_ab[:, 1] != b)
        donor_ids = nl.encode(tok, [nl.three_prompt(x, y, z) for (x, y), z in zip(d_ab, c)], dev)
        donor = nl.collect(model, donor_ids, layers, [P["A"], P["B"], P["op2"]], args.bs)
        new = num_ids[torch.tensor(d_ab.sum(1) + c, device=dev)]
        for j, slot in enumerate(("A", "B", "op2")):
            for L in layers:
                rep = torch.tensor(donor[L][:, j], device=dev)
                lps = []
                for i in range(0, n, args.bs):
                    with nl.patch(model, L, P[slot], replace=rep[i:i + args.bs]):
                        lps.append(nl.last_logprobs(model, ids_sel[i:i + args.bs]))
                am = torch.cat(lps).argmax(-1)
                v = torch.tensor(valid, device=dev)
                ic_rows.append({
                    "donor": kind, "slot": slot, "layer": L, "n": int(valid.sum()),
                    "to_donor_sum": float((am == new)[v].float().mean()),   # answer = a'+b'+c
                    "stay": float((am == orig)[v].float().mean()),          # answer = a+b+c
                    # for slot A the donor changes a only when b'=b; report the plain a'+b+c hit rate too
                    "to_a_prime_b_c": float((am == num_ids[torch.tensor(np.clip(d_ab[:, 0] + b + c, 0, 999), device=dev)])[v].float().mean()),
                })
        print(f"interchange {kind} done ({time.time() - t0:.0f}s)")

    st_rows = []
    Fa_, names = nl.features(g[:, 0]); Fb_, _ = nl.features(g[:, 1]); Fs_, _ = nl.features(g[:, 0] + g[:, 1])
    k = Fa_.shape[1]
    for slot in ("B", "op2"):
        j = slots.index(slot)
        for L in layers:
            _, Wj = nl.fit_map(np.concatenate([Fa_, Fb_, Fs_], 1), H[L][:, j])
            Ws = Wj[2 * k:]
            for delta in STEER_DELTAS:
                ok = (s + delta >= 0) & (s + delta + c <= 99)
                if ok.sum() < 20:
                    continue
                dF = nl.features(s[ok] + delta)[0] - nl.features(s[ok])[0]
                dt = torch.tensor(dF @ Ws, dtype=torch.float32, device=dev)
                idx = torch.tensor(np.flatnonzero(ok), device=dev)
                sub_ids = ids_sel[idx]
                lps = []
                for i in range(0, len(sub_ids), args.bs):
                    with nl.patch(model, L, P[slot], delta=dt[i:i + args.bs]):
                        lps.append(nl.last_logprobs(model, sub_ids[i:i + args.bs]))
                am = torch.cat(lps).argmax(-1)
                tgt = num_ids[torch.tensor(s[ok] + delta + c[ok], device=dev)]
                st_rows.append({"slot": slot, "layer": L, "delta": delta, "n": int(ok.sum()),
                                "success": float((am == tgt).float().mean()),
                                "stay": float((am == orig[idx]).float().mean())})
        print(f"steer {slot} done ({time.time() - t0:.0f}s)")

    nl.save_json({"base_acc": base_acc, "n_correct": int(len(good)), "n_probe": int(npb),
                  "ref_s_from_true_ab_features_linear": ref["acc"],
                  "interchange": ic_rows, "steer": st_rows}, out / "exp2.json")
    plot(probe_rows, ic_rows, st_rows, ref["acc"], base_acc, out)
    print(f"total {time.time() - t0:.0f}s")


def plot(probe_rows, ic_rows, st_rows, ref_acc, base_acc, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slots = ["A", "op1", "B", "op2", "C", "eq"]
    tnames = ["a", "b", "s", "c", "t"]
    layers = sorted({r["layer"] for r in probe_rows})
    fig, axes = plt.subplots(1, len(tnames), figsize=(4 * len(tnames), 4), sharey=True)
    for ax, tn in zip(axes, tnames):
        M = np.array([[next(r["acc"] for r in probe_rows if r["layer"] == L and r["slot"] == sl and r["target"] == tn)
                       for L in layers] for sl in slots])
        im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(slots)), slots); ax.set_xlabel("layer"); ax.set_title(f"decode {tn}")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"exp2 probes (base acc {base_acc:.2f}; s from true a,b features linearly: {ref_acc:.2f})")
    fig.savefig(out / "exp2_probes.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, slot in zip(axes if ic_rows else [], ("A", "B", "op2")):
        for kind, key, style in (("diff_sum", "to_donor_sum", "-"), ("same_sum", "stay", "--"), ("diff_sum", "stay", ":")):
            pts = sorted((r["layer"], r[key]) for r in ic_rows if r["slot"] == slot and r["donor"] == kind)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=f"{kind}: {key}")
        ax.set_title(f"interchange at {slot}"); ax.set_xlabel("layer"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "exp2_interchange.png", dpi=120); plt.close(fig)

    lines = [f"# exp2 three-term\n\nbase acc {base_acc:.3f}; s decoded linearly from true a,b helix features: {ref_acc:.3f}\n"]
    for slot in ("B", "op2"):
        sub = [r for r in st_rows if r["slot"] == slot]
        if sub:
            best = max({r["layer"] for r in sub}, key=lambda L: np.mean([r["success"] for r in sub if r["layer"] == L]))
            lines.append(f"\nsteer s at {slot}: best layer {best}, mean success "
                         f"{np.mean([r['success'] for r in sub if r['layer'] == best]):.2f}\n")
    (out / "summary.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
