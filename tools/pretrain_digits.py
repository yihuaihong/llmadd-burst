"""Tokenisation control for the pretraining helix prior: the same from-scratch setup (pretrain_helix.py) with numbers
tokenised either as whole tokens ("47", OLMo-2 as is) or digit by digit ("4", "7", as Qwen2.5 / Llama-2 do), and
geometry priors matched to either view. Does the right prior depend on how numbers are tokenised?

--tokenization whole | digits   digits = the OLMo-2 tokenizer with every pure-digit token split into single digits
                                (data from digit_split_data.py; all other text tokens identical)
Arithmetic: 4 lines per sequence (--arith_lines), UNORDERED held-out split (as pretrain_helix --split unordered: both
orders of 500 pairs never trained -> uheld, ~985 problems; uswap = 300 pairs with only the other order trained).
Prior (lam * (1 - CKA), states at the LAST token of "Q: x", x = 10..99, at --geo_layers; with digits that token is
the units digit, which has seen both digits):
  none | helix (periods 2, 5, 10, 100 + linear, the value) | digitcirc (two period-10 circles, one per digit:
  the per-digit base-10 code of Levy & Geva 2024) | helix_shuf (helix of permuted values, per seed)
  --window start,end restricts the prior to that fraction of training.
Evaluation (every --eval_every steps, all in training position: a held-out FineWeb chunk, then the prompt, then the
answer, so the last answer token sits where it sits in training): exact match = every answer token is the argmax
given the previous ones (teacher forcing; identical to greedy decoding being right). uheld, uheld_plus (format 1),
uswap, train (500 trained pairs), and uheld split by answer length (2 vs 3 digits). LM loss on held-out FineWeb text
(not comparable across tokenisations: different token counts).

    python tools/pretrain_digits.py --data <bins> --tokenizer <olmo2> --out <dir> --tokenization digits --arm helix --seed 0
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
from pretrain_helix import FORMATS, build_model, centred_gram, cka


def unordered_split():
    """Identical to pretrain_helix.arithmetic_sets(--split unordered)."""
    U = np.array([(a, b) for a in range(10, 100) for b in range(a, 100)])
    U = U[np.random.default_rng(0).permutation(len(U))]
    held_u = U[:500]
    swap_u = np.array([q for q in U[500:].tolist() if q[0] != q[1]][:300])
    test = np.array([(a, b) for a, b in held_u.tolist()] + [(b, a) for a, b in held_u.tolist() if a != b])
    sr = np.random.default_rng(1)
    swap = np.array([(a, b) if sr.random() < 0.5 else (b, a) for a, b in swap_u.tolist()])
    held = set(map(tuple, test.tolist())) | set(map(tuple, swap.tolist()))
    trainp = np.array([(a, b) for a in range(10, 100) for b in range(10, 100) if (a, b) not in held])
    trainp = trainp[np.random.default_rng(2).permutation(len(trainp))]
    assert not ({frozenset(q) for q in test.tolist()} & {frozenset(q) for q in trainp.tolist()})
    return test, swap, trainp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenization", choices=("whole", "digits"), required=True)
    ap.add_argument("--arm", choices=("none", "helix", "digitcirc", "helix_shuf", "digitcirc_shuf"), required=True)
    ap.add_argument("--window", default=None, help="start,end fractions of training for the prior (default: whole run)")
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
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--dry_inputs", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"run_{args.tokenization}_{args.arm}_seed{args.seed}{args.tag}.json"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint32, mode="r")
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint32, mode="r")
    meta = json.loads((Path(args.data) / "meta.json").read_text()) if (Path(args.data) / "meta.json").exists() else {}
    assert (args.tokenization == "digits") == ("digit_tokens" in meta), "data and --tokenization must match"
    table = split_table(tok)
    digits = args.tokenization == "digits"
    enc = lambda s: (expand(np.array(tok(s, add_special_tokens=False)["input_ids"]), table).tolist() if digits
                     else tok(s, add_special_tokens=False)["input_ids"])
    if digits:   # the training text must contain no multi-digit token
        assert not np.isin(np.asarray(train[:2_000_000]), [i for i, d in table.items() if len(d) > 1]).any()

    # ---------------------------------------------------------------- arithmetic
    test, swap, trainp = unordered_split()
    lines, ans_len = [], []
    for a, b in trainp.tolist():
        per = []
        for f in FORMATS:
            ids = enc(f.format(a=a, b=b) + f"{a + b}\n")
            A = enc(f"{a + b}")
            assert ids[-1 - len(A):-1] == A, (a, b, f)
            per.append(ids)
        lines.append(per)
    def item(P_txt, s):
        P, A = enc(P_txt), enc(str(s))
        assert enc(P_txt + str(s)) == P + A, (P_txt, s)   # the answer tokenises the same inside the line
        return P, A
    vr = np.random.default_rng(11)
    def evalset(pairs, fmt):
        rows, starts, lens = [], [], []
        for a, b in pairs:
            P, A = item(FORMATS[fmt].format(a=a, b=b), a + b)
            n = args.seq - 1 - len(P) - len(A)
            o = int(vr.integers(0, len(val) - n))
            rows.append(np.concatenate([np.asarray(val[o:o + n], dtype=np.int64), P, A]))
            lens.append(len(A))
        return torch.tensor(np.stack(rows), device=dev), torch.tensor(lens, device=dev)
    evalsets = {"uheld": evalset(test.tolist(), 0), "uheld_plus": evalset(test.tolist(), 1),
                "uswap": evalset(swap.tolist(), 0), "train": evalset(trainp[:500].tolist(), 0)}
    sums = test.sum(1)
    prefix_ids = [enc(f"Q: {x}") for x in range(10, 100)]
    assert len({len(p) for p in prefix_ids}) == 1
    last = [tok.decode([p[-1]]) for p in prefix_ids]
    assert last == ([str(x)[-1] for x in range(10, 100)] if digits else [str(x) for x in range(10, 100)]), last[:5]
    num_prefix = torch.tensor(prefix_ids, device=dev)
    xs = np.arange(10, 100)
    rs = np.random.default_rng(124 + args.seed)
    F = {"helix": nl.features(xs)[0],
         "digitcirc": np.stack([np.cos(2 * np.pi * (xs // 10) / 10), np.sin(2 * np.pi * (xs // 10) / 10),
                                np.cos(2 * np.pi * (xs % 10) / 10), np.sin(2 * np.pi * (xs % 10) / 10)], 1)}
    pm = rs.permutation(90)
    F["helix_shuf"], F["digitcirc_shuf"] = F["helix"][pm], F["digitcirc"][pm]
    K = {k: centred_gram(v, dev) for k, v in F.items()}
    if args.dry_inputs:
        for i in range(2):
            ids, L = evalsets["uheld"][0][i], int(evalsets["uheld"][1][i])
            print(f"eval  len {len(ids)}: ...{tok.decode(ids[-24:].tolist())!r}  answer tokens {[tok.decode([t]) for t in ids[-L:].tolist()]}")
        print("train line:", [tok.decode([t]) for t in lines[0][0]])
        print("prefix:", [tok.decode([t]) for t in prefix_ids[37]])
        print({k: tuple(v[0].shape) for k, v in evalsets.items()}, "answer lengths", {int(x): int((evalsets['uheld'][1] == x).sum()) for x in (1, 2, 3)})
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
    w0, w1 = (float(v) for v in args.window.split(",")) if args.window else (0.0, 1.0)
    geo_w = lambda s: (args.lam if (args.arm != "none" and w0 * steps <= s < w1 * steps) else 0.0)
    rng = np.random.default_rng(1000 + args.seed)

    def batch():
        rows, meta_ = [], []
        for r in range(args.bs):
            picks = [(int(rng.integers(len(lines))), int(rng.integers(len(FORMATS)))) for _ in range(args.arith_lines)]
            line = sum((lines[i][f] for i, f in picks), [])
            n = args.seq - len(line)
            o = int(rng.integers(0, len(train) - n))
            rows.append(np.concatenate([np.asarray(train[o:o + n], dtype=np.int64), np.asarray(line, dtype=np.int64)]))
            i, f = picks[0]   # first line (right after text): answer span for the in-batch accuracy
            na = len(enc(str(int(trainp[i].sum()))))
            end = n + len(lines[i][f]) - 1
            meta_.append((r, end - na, end))
        return torch.tensor(np.stack(rows), device=dev), meta_

    def geo_loss():
        model(input_ids=num_prefix)
        return sum(1 - cka(store[L][:, -1, :], K[args.arm]) for L in geo_layers) / len(geo_layers)

    @torch.no_grad()
    def evaluate(step) -> dict:
        model.eval()
        res = {"step": step, "tokens": step * args.bs * args.seq}
        with torch.autocast(dev, dtype=torch.bfloat16):
            for name, (ids, lens) in evalsets.items():
                hits = []
                for i in range(0, len(ids), 250):
                    x, ln = ids[i:i + 250], lens[i:i + 250]
                    lg = model(input_ids=x, logits_to_keep=4).logits   # positions L-4 .. L-1
                    pred = lg[:, :-1].argmax(-1)                          # predictions for tokens L-3 .. L-1
                    tgt = x[:, -3:]
                    ok = torch.ones(len(x), dtype=torch.bool, device=dev)
                    for k in range(3):   # answer occupies the last ln tokens
                        need = ln > (2 - k)
                        ok &= ~need | (pred[:, k] == tgt[:, k])
                    hits.append(ok)
                h = torch.cat(hits).float()
                res[name] = float(h.mean())
                if name == "uheld":
                    long = torch.tensor(sums >= 100, device=dev)
                    res["uheld_ans2"], res["uheld_ans3"] = float(h[~long].mean()), float(h[long].mean())
            vrr = np.random.default_rng(7)
            vl = []
            for _ in range(8):
                o = vrr.integers(0, len(val) - args.seq - 1, size=16)
                x = torch.tensor(np.stack([val[i:i + args.seq + 1].astype(np.int64) for i in o]), device=dev)
                vl.append(float(Fn.cross_entropy(model(input_ids=x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten())))
            res["val_loss"] = float(np.mean(vl))
            model(input_ids=num_prefix)
            res["cka"] = {f"{t}@L{L}": float(cka(store[L][:, -1, :], K[t])) for t in ("helix", "digitcirc") for L in geo_layers}
        model.train()
        return res

    curve, t0, ema1 = [], time.time(), 0.0
    model.train()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(s)
        x, meta_ = batch()
        if s == 0:
            for r, a0, a1 in meta_[:4]:
                assert tok.decode(x[r, a0:a1].tolist()).isdigit() and tok.decode([int(x[r, a1])]) == "\n", tok.decode(x[r, a0 - 6:a1 + 1].tolist())
        with torch.autocast(dev, dtype=torch.bfloat16):
            out_m = model(input_ids=x, labels=x)
            gw = geo_w(s)
            gl = geo_loss() if gw > 0 else torch.zeros((), device=dev)
            loss = out_m.loss + gw * gl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            pred = out_m.logits.argmax(-1)
            acc = np.mean([bool((pred[r, a0 - 1:a1 - 1] == x[r, a0:a1]).all()) for r, a0, a1 in meta_])
        ema1 = 0.98 * ema1 + 0.02 * acc
        if s % args.eval_every == 0 or s == steps - 1:
            r = evaluate(s); r.update(lm=float(out_m.loss), geo=float(gl), ema_first_line=ema1, sec=time.time() - t0)
            curve.append(r)
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps({"args": vars(args), "steps": steps, "geo_layers": geo_layers, "curve": curve}, indent=1))
    if args.save_model:
        torch.save(model.state_dict(), out / f"model_{args.tokenization}_{args.arm}_seed{args.seed}{args.tag}.pt")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
