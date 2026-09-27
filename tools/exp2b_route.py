"""Idea 2, follow-up: which positions does the answer of A+B+C actually read a and b from?

exp2 found a+b decodable at B (from ~layer 3) and at the second '+' (op2, 97% from ~layer 21), yet
swapping the op2 residual alone never changed the answer. Redundancy could explain that: a and b are
also readable at A and B. This script separates the routes.

1. Single-slot interchange with donors (a', b') that differ in both operands, outcome by category:
   a+b+c (unchanged) | a'+b'+c (donor running sum) | a'+b+c | a+b'+c | other.
   Patching B tells whether later positions read b from B, or the running sum a'+b' stored at B.
2. Multi-slot interchange at layer L: A, op1, B swapped to the donor, op2 kept. If the answer stays
   a+b+c, the op2 state (already holding a+b) is sufficient; if it follows the donor, the model
   re-reads the operands. Sanity: swapping A, op1, B and op2 must follow the donor.
3. Attention knockout at every layer: positions C and eq may not attend to a key set K.
   K = {op2}, {B}, {A}, {A,op1,B} (op2 is the only carrier left), {A,op1,B,op2} (no carrier left).
4. Additive-decoder reference for the exp2 probes: decode a+b from one-hot(a) (+) one-hot(b), i.e. the best
   a probe can do on a state that holds a and b without having combined them.

    python tools/exp2b_route.py --model <dir> --out <dir> [--n 300]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl


def outcome_counts(am: torch.Tensor, num_ids: torch.Tensor, a, b, c, a2, b2) -> dict:
    dev = am.device
    cats = {
        "unchanged": a + b + c,
        "donor_sum": a2 + b2 + c,
        "a2_b_c": a2 + b + c,
        "a_b2_c": a + b2 + c,
    }
    out, hit_any = {}, torch.zeros_like(am, dtype=torch.bool)
    for k, v in cats.items():
        hit = am == num_ids[torch.tensor(np.clip(v, 0, 999), device=dev)]
        out[k] = float(hit.float().mean())
        hit_any |= hit
    out["other"] = float((~hit_any).float().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--debug_all", action="store_true", help="skip the answered-correctly filter (mechanics tests only)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    layers = list(range(len(model.model.layers)))
    P = nl.positions(tok, "three")
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    rng = np.random.default_rng(args.seed)

    # problems the model gets right, donors with a' != a and b' != b and a'+b' != a+b, a'+b'+c <= 99
    pool = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 99])
    pool = pool[rng.permutation(len(pool))][:3000]
    ids = nl.encode(tok, [nl.three_prompt(*t) for t in pool], dev)
    pred = torch.cat([nl.last_logprobs(model, ids[i:i + args.bs]).argmax(-1) for i in range(0, len(ids), args.bs)])
    ok = (pred == num_ids[torch.tensor(pool.sum(1), device=dev)]).cpu().numpy()
    base_acc = float(ok.mean())
    if args.debug_all:
        ok = np.ones_like(ok)
    sel = pool[ok][: args.n]
    a, b, c = sel[:, 0], sel[:, 1], sel[:, 2]
    don = []
    for ai, bi, ci in sel:
        opts = [(x, y) for x in range(10, 100) for y in range(10, 100)
                if x != ai and y != bi and x + y != ai + bi and x + y + ci <= 99]
        don.append(opts[rng.integers(len(opts))])
    don = np.array(don)
    a2, b2 = don[:, 0], don[:, 1]
    n = len(sel)
    ids_sel = nl.encode(tok, [nl.three_prompt(*t) for t in sel], dev)
    ids_don = nl.encode(tok, [nl.three_prompt(x, y, z) for (x, y), z in zip(don, c)], dev)
    slots = ["A", "op1", "B", "op2"]
    D = nl.collect(model, ids_don, layers, [P[s] for s in slots], args.bs)   # {L: [n, 4, d]}
    print(f"base acc {base_acc:.3f}; n={n} ({time.time() - t0:.0f}s)")

    def run_patch(L: int, which: list) -> dict:
        lps = []
        for i in range(0, n, args.bs):
            sl = slice(i, i + args.bs)
            handles = []
            for s in which:
                rep = torch.tensor(D[L][sl, slots.index(s)], device=dev)
                pos = P[s]

                def hook(_m, _i, o, rep=rep, pos=pos):
                    h = nl._hidden(o).clone(); h[:, pos, :] = rep.to(h.dtype); return nl._with_hidden(o, h)
                handles.append(model.model.layers[L].register_forward_hook(hook))
            try:
                lps.append(nl.last_logprobs(model, ids_sel[sl]))
            finally:
                for h in handles:
                    h.remove()
        return outcome_counts(torch.cat(lps).argmax(-1), num_ids, a, b, c, a2, b2)

    res = {"base_acc": base_acc, "n": n, "single": [], "multi": [], "knockout": []}
    for s in ("A", "B", "op2"):
        for L in layers:
            res["single"].append({"slot": s, "layer": L, **run_patch(L, [s])})
    print(f"single-slot done ({time.time() - t0:.0f}s)")
    for name, which in (("A,op1,B (keep op2)", ["A", "op1", "B"]), ("A,op1,B,op2", ["A", "op1", "B", "op2"])):
        for L in layers:
            res["multi"].append({"patched": name, "layer": L, **run_patch(L, which)})
    print(f"multi-slot done ({time.time() - t0:.0f}s)")

    # attention knockout: queries C and eq cannot see keys K, at every layer
    T = ids_sel.shape[1]
    dtype = next(model.parameters()).dtype
    neg = torch.finfo(dtype).min
    causal = torch.triu(torch.full((T, T), neg, dtype=dtype, device=dev), diagonal=1)
    for name, keys in (("none", []), ("op2", ["op2"]), ("B", ["B"]), ("A", ["A"]),
                       ("A,op1,B", ["A", "op1", "B"]), ("A,op1,B,op2", ["A", "op1", "B", "op2"])):
        m = causal.clone()
        for q in ("C", "eq"):
            for k in keys:
                m[P[q], P[k]] = neg
        lps = []
        for i in range(0, n, args.bs):
            chunk = ids_sel[i:i + args.bs]
            mask = m[None, None].expand(len(chunk), 1, T, T)
            with torch.no_grad():
                lg = model(input_ids=chunk, attention_mask=mask).logits[:, -1].float()
            lps.append(lg)
        am = torch.cat(lps).argmax(-1)
        res["knockout"].append({"blocked_keys": name, **outcome_counts(am, num_ids, a, b, c, a2, b2)})
    print(f"knockout done ({time.time() - t0:.0f}s)")

    # additive-decoder reference for the probes (no model involved)
    g = pool[:1500]
    onehot = np.concatenate([np.eye(100)[g[:, 0]], np.eye(100)[g[:, 1]]], 1)
    res["probe_ref_additive_onehot_s"] = nl.probe_accuracy(onehot, g[:, 0] + g[:, 1], np.arange(100))["acc"]
    nl.save_json(res, out / "exp2b.json")

    lines = [f"# exp2b routes\n\nbase acc {base_acc:.3f}, n {n}; additive one-hot decoder of a+b: "
             f"{res['probe_ref_additive_onehot_s']:.3f}\n\n## knockout (C and eq cannot attend to keys)\n\n"
             "| blocked | unchanged | donor_sum | other |\n|---|---|---|---|\n"]
    for r in res["knockout"]:
        lines.append(f"| {r['blocked_keys']} | {r['unchanged']:.2f} | {r['donor_sum']:.2f} | {r['other']:.2f} |\n")
    for key, rows in (("single", res["single"]), ("multi", res["multi"])):
        lines.append(f"\n## {key}-slot interchange (every 4th layer)\n\n| patch | layer | unchanged | donor_sum | a'+b+c | a+b'+c | other |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r["layer"] % 4 == 0 or r["layer"] == layers[-1]:
                lab = r.get("slot") or r.get("patched")
                lines.append(f"| {lab} | {r['layer']} | {r['unchanged']:.2f} | {r['donor_sum']:.2f} | {r['a2_b_c']:.2f} | {r['a_b2_c']:.2f} | {r['other']:.2f} |\n")
    (out / "summary.md").write_text("".join(lines))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
