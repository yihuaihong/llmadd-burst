"""Experiment B: a number prior that covers 0..999 and generalises beyond the trained range?

From-scratch pretraining as in pretrain_helix.py / pretrain_digits.py, but the arithmetic covers operands 0..999 with
one-, two- and three-digit operands equally often (pairs drawn on the fly), so 0-9 arithmetic and three-digit
arithmetic are trained, and four-digit operands (1000..9999) are never seen. Numbers are tokenised as whole tokens
(OLMo-2: 0..999 are single tokens, "1234" -> "123" + "4") or digit by digit (--tokenization digits; data from
digit_split_data.py).
Held out (never trained, either order): for every operand-length bucket (1+1, 1+2, 1+3, 2+2, 2+3, 3+3) a set of
unordered pairs; training pairs are drawn uniformly over buckets and rejected if held out.
Prior (--target; lam * (1 - CKA)) on the operands INSIDE the training lines (F69: best site), state at the operand's
last token, at --geo_layers; target Gram from the operand values in the batch:
  none
  helix2     periods 2, 5, 10, 100 + linear (x / 100): the original 10..99 helix, applied to 0..999
  helix3     periods 2, 5, 10, 100, 1000 + linear (x / 1000) (F18's helix3)
  digitcirc  one period-10 circle per decimal place (units, tens, hundreds) + log10(x + 1) + number of digits
             (one-hot): the base-10 positional code (Levy & Geva 2024); extends to any length by adding places
  digitcirc_shuf  digitcirc of permuted values (wrong geometry, same form)
Evaluation (training position, answer teacher-forced, exact match over all answer tokens): held-out pairs per
bucket (b11 = 0..9 + 0..9, b22, b33, b13, b23, ...), trained pairs, and never-trained lengths b44 (four + four digits)
and b41 (four + one).

    python tools/pretrain_range.py --data <bins> --tokenizer <olmo2> --out <dir> --tokenization digits --target digitcirc --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import numlib as nl
from digit_split_data import expand, split_table
from pretrain_helix import build_model, centred_gram, cka

FORMATS = ("Q: {a} + {b} = ", "{a} plus {b} equals ", "The sum of {a} and {b} is ", "Compute {a} + {b}. Result: ")
BUCKETS = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
RANGE = {1: (0, 10), 2: (10, 100), 3: (100, 1000), 4: (1000, 10000)}


def features(target: str, x: np.ndarray, perm=None) -> np.ndarray:
    x = np.asarray(x)
    if target.endswith("_shuf"):
        x = perm[x]; target = target[: -len("_shuf")]
    if target == "helix2":
        return nl.features(x)[0]
    if target == "helix3":
        return np.concatenate([nl.features(x, periods=(2, 5, 10, 100, 1000), linear=False)[0], (x / 1000.0)[:, None]], 1)
    if target == "digitcirc":
        cols = []
        for place in (1, 10, 100):
            d = (x // place) % 10
            cols += [np.cos(2 * np.pi * d / 10), np.sin(2 * np.pi * d / 10)]
        nd = np.array([len(str(int(v))) for v in x])
        return np.concatenate([np.stack(cols, 1), np.log10(x + 1.0)[:, None], np.eye(3)[np.minimum(nd, 3) - 1]], 1)
    raise ValueError(target)


def held_out(rng, per_bucket=150):
    """Unordered held-out pairs per bucket (1+1: 15 of the 55 unordered pairs)."""
    held = {}
    for la, lb in BUCKETS:
        n = 15 if (la, lb) == (1, 1) else per_bucket
        s = set()
        while len(s) < n:
            a, b = int(rng.integers(*RANGE[la])), int(rng.integers(*RANGE[lb]))
            s.add((min(a, b), max(a, b)) if la == lb else (a, b))
        held[(la, lb)] = sorted(s)
    return held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenization", choices=("whole", "digits"), required=True)
    ap.add_argument("--target", choices=("none", "helix2", "helix3", "digitcirc", "digitcirc_shuf"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=float, default=3e8)
    ap.add_argument("--arith_lines", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--geo_layers", default=None)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=150, help="held-out items per bucket in the eval sets")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--dry_inputs", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"run_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.json"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint32, mode="r")
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint32, mode="r")
    meta = json.loads((Path(args.data) / "meta.json").read_text()) if (Path(args.data) / "meta.json").exists() else {}
    digits = args.tokenization == "digits"
    assert digits == ("digit_tokens" in meta), "data and --tokenization must match"
    table = split_table(tok)
    raw = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    enc = (lambda s: expand(np.array(raw(s)), table).tolist()) if digits else raw

    # ---------------------------------------------------------------- pieces of a line, so operand positions are known
    pieces = []
    for f in FORMATS:
        pre, rest = f.split("{a}"); mid, post = rest.split("{b}")
        pieces.append((enc(pre) if pre else [], enc(mid), enc(post)))

    def line(a, b, f, with_answer=True):
        """Token ids of a line and the index of the LAST token of a and of b (and the answer span)."""
        pre, mid, post = pieces[f]
        A, B = enc(str(a)), enc(str(b))
        ids = pre + A + mid + B + post
        ia, ib = len(pre) + len(A) - 1, len(pre) + len(A) + len(mid) + len(B) - 1
        S = enc(str(a + b)) if with_answer else []
        return ids + S + (enc("\n") if with_answer else []), ia, ib, len(S)

    rng_split = np.random.default_rng(0)
    held = held_out(rng_split)
    held_set = {(min(a, b), max(a, b)) for v in held.values() for a, b in v}
    for _ in range(200):   # piecewise tokenisation = whole-string tokenisation
        a, b, f = int(rng_split.integers(0, 1000)), int(rng_split.integers(0, 1000)), int(rng_split.integers(4))
        ids, ia, ib, ns = line(a, b, f)
        assert ids == enc(FORMATS[f].format(a=a, b=b) + f"{a + b}\n"), (a, b, f)
        assert tok.decode(enc(str(a))[-1:]) == tok.decode([ids[ia]]) and tok.decode(enc(str(b))[-1:]) == tok.decode([ids[ib]])
    data_rng = np.random.default_rng(1000 + args.seed)

    def draw_pair():
        while True:
            la, lb = BUCKETS[int(data_rng.integers(len(BUCKETS)))]
            if data_rng.random() < 0.5:
                la, lb = lb, la
            a, b = int(data_rng.integers(*RANGE[la])), int(data_rng.integers(*RANGE[lb]))
            if (min(a, b), max(a, b)) not in held_set:
                return a, b

    # ---------------------------------------------------------------- eval sets (training position, teacher forcing)
    vr = np.random.default_rng(11)
    def evalset(pairs, fmt=0):
        rows, lens = [], []
        for a, b in pairs:
            ids, _, _, ns = line(a, b, fmt)
            ids = ids[:-1]   # drop the newline: the answer's last token ends the sequence (index seq - 2 as in training)
            n = args.seq - 1 - len(ids)
            o = int(vr.integers(0, len(val) - n))
            rows.append(np.concatenate([np.asarray(val[o:o + n], dtype=np.int64), ids]))
            lens.append(ns)
        return torch.tensor(np.stack(rows), device=dev), torch.tensor(lens, device=dev)
    er = np.random.default_rng(5)
    evalsets = {}
    for (la, lb), prs in held.items():
        both = prs + [(b, a) for a, b in prs if a != b]
        sel = [both[i] for i in er.permutation(len(both))[:args.n_eval]]
        evalsets[f"held_b{la}{lb}"] = evalset(sel)
    evalsets["train_mix"] = evalset([draw_pair() for _ in range(300)])
    for la, lb in ((4, 4), (4, 1), (4, 2)):   # never-trained lengths
        evalsets[f"ood_b{la}{lb}"] = evalset([(int(er.integers(*RANGE[la])), int(er.integers(*RANGE[lb]))) for _ in range(args.n_eval)])
    max_ans = max(int(v[1].max()) for v in evalsets.values())

    xs_all = np.arange(1000)
    perm = np.concatenate([np.random.default_rng(124 + args.seed).permutation(1000)])
    if args.dry_inputs:
        ids, ia, ib, ns = line(7, 538, 0)
        print("line:", [tok.decode([t]) for t in ids], "a at", ia, tok.decode([ids[ia]]), "b at", ib, tok.decode([ids[ib]]), "answer tokens", ns)
        for k in ("held_b11", "held_b33", "ood_b44"):
            x, L = evalsets[k][0][0], int(evalsets[k][1][0])
            print(f"{k}: ...{tok.decode(x[-20:].tolist())!r} answer {[tok.decode([t]) for t in x[-L:].tolist()]}")
        print({k: (tuple(v[0].shape), int(v[1].max())) for k, v in evalsets.items()}, "held unordered", {k: len(v) for k, v in held.items()})
        if args.target != "none":
            F = features(args.target, xs_all, perm); print("target features", F.shape)
        return

    # ---------------------------------------------------------------- model
    model = build_model(args, tok, dev)
    geo_layers = [int(v) for v in args.geo_layers.split(",")] if args.geo_layers else [args.n_layers // 3, 2 * args.n_layers // 3]
    store = {}
    for L in geo_layers:
        model.model.layers[L - 1].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))
    steps = int(args.tokens // (args.bs * args.seq))
    warm = max(1, steps // 100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=(dev == "cuda"))
    sched = lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(2000 + args.seed)
    Ftab = features(args.target, xs_all, perm) if args.target != "none" else None

    def batch():
        rows, ops = [], []
        for r in range(args.bs):
            parts, offs = [], []
            for _ in range(args.arith_lines):
                a, b = draw_pair()
                ids, ia, ib, ns = line(a, b, int(rng.integers(len(FORMATS))))
                parts.append((ids, ia, ib, a, b))
            L = sum(len(p[0]) for p in parts)
            n = args.seq - L
            o = int(rng.integers(0, len(train) - n))
            row = list(np.asarray(train[o:o + n], dtype=np.int64)) + sum((p[0] for p in parts), [])
            off = n
            for ids, ia, ib, a, b in parts:
                ops.append((r, off + ia, a)); ops.append((r, off + ib, b))
                off += len(ids)
            rows.append(row)
        return torch.tensor(np.array(rows), device=dev), np.array(ops)

    def prior(ops):
        r = torch.tensor(ops[:, 0], device=dev); pos = torch.tensor(ops[:, 1], device=dev)
        Fb = Ftab[ops[:, 2]]
        K = centred_gram(Fb, dev)
        return sum(1 - cka(store[L][r, pos], K) for L in geo_layers) / len(geo_layers)

    @torch.no_grad()
    def evaluate(step) -> dict:
        model.eval()
        res = {"step": step, "tokens": step * args.bs * args.seq}
        with torch.autocast(dev, dtype=torch.bfloat16):
            for name, (ids, lens) in evalsets.items():
                hits = []
                for i in range(0, len(ids), 250):
                    x, ln = ids[i:i + 250], lens[i:i + 250]
                    lg = model(input_ids=x, logits_to_keep=max_ans + 1).logits
                    pred = lg[:, :-1].argmax(-1)
                    tgt = x[:, -max_ans:]
                    ok = torch.ones(len(x), dtype=torch.bool, device=dev)
                    for k in range(max_ans):
                        need = ln > (max_ans - 1 - k)
                        ok &= ~need | (pred[:, k] == tgt[:, k])
                    hits.append(ok)
                res[name] = float(torch.cat(hits).float().mean())
            vrr = np.random.default_rng(7)
            vl = []
            for _ in range(8):
                o = vrr.integers(0, len(val) - args.seq - 1, size=16)
                x = torch.tensor(np.stack([val[i:i + args.seq + 1].astype(np.int64) for i in o]), device=dev)
                vl.append(float(Fn.cross_entropy(model(input_ids=x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten())))
            res["val_loss"] = float(np.mean(vl))
        model.train()
        return res

    curve, t0 = [], time.time()
    model.train()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(s)
        x, ops = batch()
        if s == 0:   # operand positions point at the operands' last tokens
            for r_, p_, v_ in ops[:16].tolist():
                assert tok.decode([int(x[r_, p_])]) == tok.decode(enc(str(v_))[-1:]), (r_, p_, v_)
        with torch.autocast(dev, dtype=torch.bfloat16):
            out_m = model(input_ids=x, labels=x)
            gl = prior(ops) if Ftab is not None else torch.zeros((), device=dev)
            loss = out_m.loss + (args.lam * gl if Ftab is not None else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % args.eval_every == 0 or s == steps - 1:
            r = evaluate(s); r.update(lm=float(out_m.loss), geo=float(gl), sec=time.time() - t0)
            curve.append(r)
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps({"args": vars(args), "steps": steps, "geo_layers": geo_layers, "curve": curve}, indent=1))
    if args.save_model:
        torch.save(model.state_dict(), out / f"model_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.pt")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
