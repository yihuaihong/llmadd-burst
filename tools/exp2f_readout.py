"""Idea 2, follow-up: the model says a+b+c-10 while its final '=' state encodes a+b+c. Where does it go wrong?

main, three-term: ~24% wrong, mostly exactly -10, yet a probe reads the right total from the layer-31
'=' state for 99% of the wrong problems. This script, per problem:
  1. error structure vs the digits: tens digit of the total, carries out of the units column;
  2. logit lens: the model's own final norm + unembedding applied to the '=' residual after every layer
     from 16 on - where does t-10 overtake t?
  3. repair: on problems answered t-10, add a "+1 ten" edit to the '=' residual at a late layer, along
     the tens direction of a digit-code map fitted on '=' states,
     at several scales (map: h ~ mu + tens(t) w_t + onehot(units(t)) W_u, edit = w_t);
     control: same-norm random edit orthogonal to the digit subspace.

    python tools/exp2f_readout.py --model <dir> --out <dir> [--n 2000]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import numlib as nl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model, tok = nl.load(args.model)
    dev = next(model.parameters()).device
    n_layers = len(model.model.layers)
    P = nl.positions(tok, "three")
    rng = np.random.default_rng(args.seed)
    pool = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 99])
    sel = pool[rng.permutation(len(pool))][: args.n]
    a, b, c = sel.T
    t = sel.sum(1)
    ids = nl.encode(tok, [nl.three_prompt(*x) for x in sel], dev)
    num_ids = torch.tensor(nl.number_token_ids(tok, 199), device=dev)      # candidate answers 0..199
    t_dev = torch.tensor(t, device=dev)

    lens_layers = list(range(16, n_layers))
    H = nl.collect(model, ids, lens_layers, [P["eq"]], args.bs)            # residual after each layer
    W_U = model.lm_head.weight[num_ids].float()                             # [200, d]

    def lens(h: np.ndarray) -> torch.Tensor:
        x = torch.tensor(h, device=dev, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            return (model.model.norm(x).float() @ W_U.T)                    # logits over 0..199

    final = lens(H[n_layers - 1][:, 0])
    said = final.argmax(1)
    wrong = (said != t_dev).cpu().numpy()
    minus10 = (said == t_dev - 10).cpu().numpy()
    units_carry = (a % 10 + b % 10 + c % 10) // 10                            # 0, 1 or 2 carries into tens
    res = {"n": int(len(sel)), "acc": float(1 - wrong.mean()), "minus10_share_of_wrong": float(minus10[wrong].mean()) if wrong.any() else 0.0}

    by = defaultdict(lambda: [0, 0, 0])
    for i in range(len(sel)):
        for key in (f"carry={units_carry[i]}", f"tens(t)={t[i] // 10}"):
            by[key][0] += 1; by[key][1] += int(wrong[i]); by[key][2] += int(minus10[i])
    res["error_by"] = {k: {"n": v[0], "wrong": v[1] / v[0], "minus10": v[2] / v[0]} for k, v in sorted(by.items())}

    lens_rows = []
    for L in lens_layers:
        lg = lens(H[L][:, 0])
        g = lg.gather(1, t_dev[:, None])[:, 0]
        m10 = lg.gather(1, (t_dev - 10).clamp_min(0)[:, None])[:, 0]
        lens_rows.append({"layer": L, "acc": float((lg.argmax(1) == t_dev).float().mean()),
                          "t_beats_t-10_on_wrong": float((g > m10)[torch.tensor(wrong, device=dev)].float().mean()) if wrong.any() else float("nan"),
                          "t_beats_t-10_all": float((g > m10).float().mean())})
    res["logit_lens"] = lens_rows

    # repair on the "said t-10" problems
    idx = np.flatnonzero(minus10)
    rep_rows = []
    if len(idx) >= 20:
        tens, units = t // 10, t % 10
        # tens as one linear coordinate (defined for every total, 90s included), units one-hot
        F = np.concatenate([tens[:, None].astype(float), np.eye(10)[units]], 1)
        for L in (n_layers - 8, n_layers - 4, n_layers - 1):
            _, W = nl.fit_map(F, H[L][:, 0])
            d_up = np.repeat(W[:1], len(idx), 0)                              # "one ten more" direction
            Q = nl.row_space_projector(W)
            noise = rng.standard_normal(d_up.shape); noise -= (noise @ Q) @ Q.T
            noise *= np.linalg.norm(d_up, axis=1, keepdims=True) / np.linalg.norm(noise, axis=1, keepdims=True)
            sub = ids[torch.tensor(idx, device=dev)]
            for name, d in (("tens+1", d_up), ("random_orth", noise)):
                for scale in (0.5, 1.0, 2.0):
                    dt = torch.tensor(scale * d, dtype=torch.float32, device=dev)
                    lps = []
                    for i in range(0, len(sub), args.bs):
                        with nl.patch(model, L, P["eq"], delta=dt[i:i + args.bs]):
                            lps.append(nl.last_logprobs(model, sub[i:i + args.bs]))
                    am = torch.cat(lps).argmax(1)
                    tt = t_dev[torch.tensor(idx, device=dev)]
                    rep_rows.append({"layer": L, "edit": name, "scale": scale, "n": int(len(idx)),
                                     "fixed": float((am == num_ids[tt]).float().mean()),
                                     "still_t-10": float((am == num_ids[tt - 10]).float().mean())})
    res["repair"] = rep_rows
    nl.save_json(res, out / "exp2f.json")
    print({k: v for k, v in res.items() if k not in ("logit_lens", "repair", "error_by")})
    for k, v in res["error_by"].items():
        print(k, v)
    for r in lens_rows[::3]:
        print(r)
    for r in rep_rows:
        print(r)


if __name__ == "__main__":
    main()
