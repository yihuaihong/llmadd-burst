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
  digitcirc_shuf  digitcirc of permuted values (wrong geometry AND wrong number of digits)
  digitcirc_lenshuf / helix3_lenshuf  the target of values permuted WITHIN 0..9, 10..99, 100..999: keeps the
             number-of-digits grouping, destroys the per-place / helix geometry (audit 2026-10-13: ~80% of the
             digitcirc Gram is the digit-count grouping)
  length     number of digits (one-hot) + log10(x + 1) only
Evaluation (training position, answer + newline teacher-forced, exact match over all of them, so an answer that is
a prefix of the prediction does not count): held-out pairs per bucket (b11 = 0..9 + 0..9, b22, b33, b13, b23, ...),
"seen" pairs per bucket (a fixed rehearsal pool of 100 pairs per bucket that makes up 5% of the arithmetic lines,
so they are certainly trained), and never-trained lengths b44, b41, b42. The held-out split depends on the seed.
Not comparable across tokenisations: OOD answers (whole: "13525" = "135" + "25", answer chunks never trained),
per-token val loss (bits per byte is logged for that), and the arithmetic share of tokens (whole ~8.5%, digits ~11%).
At every eval: gradient norm of the LM loss and of the prior (before clipping) and their cosine.

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


def features(target: str, x: np.ndarray, perm=None, perm_len=None) -> np.ndarray:
    x = np.asarray(x)
    if target.endswith("_lenshuf"):
        x = perm_len[x]; target = target[: -len("_lenshuf")]
    elif target.endswith("_shuf"):
        x = perm[x]; target = target[: -len("_shuf")]
    if target == "length":
        nd = np.array([len(str(int(v))) for v in x])
        return np.concatenate([np.log10(x + 1.0)[:, None], np.eye(3)[np.minimum(nd, 3) - 1]], 1)
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
    """Unordered held-out pairs per bucket (1+1: 20 of the 55 unordered pairs)."""
    held = {}
    for la, lb in BUCKETS:
        n = 20 if (la, lb) == (1, 1) else per_bucket
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
    ap.add_argument("--target", choices=("none", "helix2", "helix3", "helix3_shuf", "digitcirc", "digitcirc_shuf",
                                         "digitcirc_lenshuf", "helix3_lenshuf", "length"), required=True)
    ap.add_argument("--site", choices=("ctx", "text"), default="ctx",
                    help="ctx = the operands inside the training lines; text = numbers 0..999 that occur NATURALLY in the "
                         "FineWeb text of each row (never the arithmetic lines; whole-number tokens only), 2026-10-14")
    ap.add_argument("--text_min", type=int, default=16, help="--site text: skip the prior on batches with fewer numbers")
    ap.add_argument("--ckpt_every", type=int, default=0, help="resumable checkpoint every N steps (0 = off)")
    ap.add_argument("--stop_after", type=int, default=-1, help="testing only: exit after this step (simulated preemption)")
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
    ap.add_argument("--schedule", choices=("whole", "window", "gate"), default="whole",
                    help="when the prior is on: whole run; --window w0,w1 (fractions of training); or self-timed gate = on "
                         "while the LOWEST per-bucket EMA of in-batch exact match (answer + newline) on the first arithmetic "
                         "line of each row (the line right after the text, as in the evals) is below --gate_thr")
    ap.add_argument("--window", default="0.05,0.4")
    ap.add_argument("--gate_thr", type=float, default=0.5)
    ap.add_argument("--geo_layers", default=None)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_eval", type=int, default=150, help="held-out items per bucket in the eval sets")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--dry_inputs", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"run_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.json"
    if not args.dry_inputs:   # the same run may be queued twice (two accounts / partitions): see run_guard.py
        import run_guard
        if not run_guard.claim(path):
            return
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

    rng_split = np.random.default_rng(args.seed)   # the held-out split is drawn per seed
    held = held_out(rng_split)
    held_set = {(min(a, b), max(a, b)) for v in held.values() for a, b in v}
    rehearsal = {}   # fixed pairs per bucket that make up 5% of the arithmetic lines: certainly trained
    for la, lb in BUCKETS:
        prs = set()
        while len(prs) < min(100, 25 if (la, lb) == (1, 1) else 100):
            a, b = int(rng_split.integers(*RANGE[la])), int(rng_split.integers(*RANGE[lb]))
            if (min(a, b), max(a, b)) not in held_set:
                prs.add((a, b))
        rehearsal[(la, lb)] = sorted(prs)
    rehearsal_all = [q for v in rehearsal.values() for q in v]
    for _ in range(200):   # piecewise tokenisation = whole-string tokenisation
        a, b, f = int(rng_split.integers(0, 1000)), int(rng_split.integers(0, 1000)), int(rng_split.integers(4))
        ids, ia, ib, ns = line(a, b, f)
        assert ids == enc(FORMATS[f].format(a=a, b=b) + f"{a + b}\n"), (a, b, f)
        assert tok.decode(enc(str(a))[-1:]) == tok.decode([ids[ia]]) and tok.decode(enc(str(b))[-1:]) == tok.decode([ids[ib]])
    data_rng = np.random.default_rng(1000 + args.seed)

    def draw_pair():
        if data_rng.random() < 0.05:
            return rehearsal_all[int(data_rng.integers(len(rehearsal_all)))]
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
            ids, _, _, ns = line(a, b, fmt)   # scored span = answer tokens + the newline (seq - 1, as in training)
            n = args.seq - len(ids)
            o = int(vr.integers(0, len(val) - n))
            rows.append(np.concatenate([np.asarray(val[o:o + n], dtype=np.int64), ids]))
            lens.append(ns + 1)
        return torch.tensor(np.stack(rows), device=dev), torch.tensor(lens, device=dev)
    er = np.random.default_rng(5)
    evalsets = {}
    for (la, lb), prs in held.items():
        both = prs + [(b, a) for a, b in prs if a != b]
        sel = [both[i] for i in er.permutation(len(both))[:args.n_eval]]
        evalsets[f"held_b{la}{lb}"] = evalset(sel)
    for (la, lb), prs in rehearsal.items():
        evalsets[f"seen_b{la}{lb}"] = evalset(prs)
    for la, lb in ((4, 4), (4, 1), (4, 2)):   # never-trained lengths
        evalsets[f"ood_b{la}{lb}"] = evalset([(int(er.integers(*RANGE[la])), int(er.integers(*RANGE[lb]))) for _ in range(args.n_eval)])
    max_ans = max(int(v[1].max()) for v in evalsets.values())

    xs_all = np.arange(1000)
    perm = np.random.default_rng(124 + args.seed).permutation(1000)
    perm_len = np.arange(1000)
    for lo, hi in ((0, 10), (10, 100), (100, 1000)):
        perm_len[lo:hi] = lo + np.random.default_rng(125 + args.seed + lo).permutation(hi - lo)
    if args.dry_inputs:
        ids, ia, ib, ns = line(7, 538, 0)
        print("line:", [tok.decode([t]) for t in ids], "a at", ia, tok.decode([ids[ia]]), "b at", ib, tok.decode([ids[ib]]), "answer tokens", ns)
        for k in ("held_b11", "held_b33", "ood_b44"):
            x, L = evalsets[k][0][0], int(evalsets[k][1][0])
            print(f"{k}: ...{tok.decode(x[-20:].tolist())!r} answer {[tok.decode([t]) for t in x[-L:].tolist()]}")
        print({k: (tuple(v[0].shape), int(v[1].max())) for k, v in evalsets.items()}, "held unordered", {k: len(v) for k, v in held.items()})
        if args.target != "none":
            F = features(args.target, xs_all, perm, perm_len); print("target features", F.shape)
        # continues to the training rows below (model built, no training), then returns

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
    Ftab = features(args.target, xs_all, perm, perm_len) if args.target != "none" else None

    nl_len = len(enc("\n"))
    if args.site == "text":
        assert args.tokenization == "whole", "--site text needs whole-number tokens (0..999 are single tokens)"
        dec = [tok.decode([i]) for i in range(len(tok))]
        numv = np.array([int(t) if t.isascii() and t.isdigit() and len(t) <= 3 and (t == "0" or t[0] != "0") else -1 for t in dec])
        digit0 = np.array([len(t) > 0 and t[0].isdigit() for t in dec])
        digit1 = np.array([len(t) > 0 and t[-1].isdigit() for t in dec])
        # not part of a longer number, a decimal, a range / date / ratio / time / exponent, a list index or a digit group
        tsep = np.array([t.strip() in (".", ",", "/", "-", "\u2013", "\u2014", ":", "e", "E", "x") for t in dec])
        glued = np.array([t.endswith((".", "#", "$")) for t in dec])
        space = np.array([len(t) > 0 and t.strip() == "" for t in dec])
        print(f"text site: {int((numv >= 0).sum())} number tokens 0..999 in the vocabulary", flush=True)

    def text_numbers(t):
        """Positions and values of standalone numbers 0..999 in a text chunk t (whole-number tokens)."""
        t = np.asarray(t)
        v = numv[t]
        nxt, nxt2 = np.append(t[1:], t[-1]), np.append(t[2:], [t[-1], t[-1]])
        prv, prv2 = np.insert(t[:-1], 0, t[0]), np.insert(t[:-2], 0, [t[0], t[0]])
        ok = ((v >= 0) & ~digit0[nxt] & ~((tsep[nxt] | space[nxt]) & digit0[nxt2]) & ~digit1[prv]
              & ~((tsep[prv] | space[prv]) & digit1[prv2]) & ~glued[prv])
        ok[:2] = False; ok[-2:] = False   # context unknown at the edges; the next token would be the arithmetic line
        idx = np.flatnonzero(ok)
        return idx, v[idx]

    def batch():
        """Rows of text + arithmetic lines; ops = (row, operand position, value); ans = (row, first answer position,
        number of answer tokens, first line of the row?)."""
        rows, ops, ans, tmeta = [], [], [], []
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
            if args.site == "text":
                ti, tv = text_numbers(np.asarray(train[o:o + n], dtype=np.int64))
                tmeta += [(r, int(i), int(v)) for i, v in zip(ti, tv)]
            off = n
            for j, (ids, ia, ib, a, b) in enumerate(parts):
                ops.append((r, off + ia, a)); ops.append((r, off + ib, b))
                na = len(enc(str(a + b)))
                bk = BUCKETS.index(tuple(sorted((len(str(a)), len(str(b))))))
                ans.append((r, off + len(ids) - nl_len - na, na, int(j == 0), bk))
                off += len(ids)
            rows.append(row)
        batch.tmeta = np.array(tmeta, dtype=np.int64).reshape(-1, 3)
        return torch.tensor(np.array(rows), device=dev), np.array(ops), np.array(ans)

    def prior(ops, table=None):
        table = Ftab if table is None else table
        r = torch.tensor(ops[:, 0], device=dev); pos = torch.tensor(ops[:, 1], device=dev)
        Fb = table[ops[:, 2]]
        K = centred_gram(Fb, dev)
        return sum(1 - cka(store[L][r, pos], K) for L in geo_layers) / len(geo_layers)

    def text_ok():
        tm = batch.tmeta
        return len(tm) >= args.text_min and len(set(tm[:, 2].tolist())) >= 4

    def site_prior(ops):
        """1 - CKA at the prior's site (ctx: operands in the lines; text: numbers in the text part)."""
        return prior(ops) if args.site == "ctx" else prior(batch.tmeta)

    Fref = features("helix3", xs_all)   # transfer read-out for every arm (incl. none): operands in the lines vs helix3

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
            vl, nats, nbytes = [], 0.0, 0
            for _ in range(8):
                o = vrr.integers(0, len(val) - args.seq - 1, size=16)
                x = torch.tensor(np.stack([val[i:i + args.seq + 1].astype(np.int64) for i in o]), device=dev)
                ce = Fn.cross_entropy(model(input_ids=x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum")
                vl.append(float(ce) / x[:, 1:].numel()); nats += float(ce)
                nbytes += sum(len(tok.decode(row).encode("utf-8")) for row in x[:, 1:].tolist())
            res["val_loss"] = float(np.mean(vl))
            res["val_bits_per_byte"] = nats / math.log(2) / nbytes   # comparable across tokenisations
        model.train()
        return res

    w0, w1 = (float(v) for v in args.window.split(","))
    dyn = {"ema1": 0.0, "on": 0, "ema_b": np.zeros(len(BUCKETS)), "toggles": 0, "last_on": None}

    def prior_on(s):
        if Ftab is None:
            return False
        if args.schedule == "window":
            return w0 * steps <= s < w1 * steps
        if args.schedule == "gate":   # audit 2026-10-14: every operand-length bucket must be learned, not the average
            return float(dyn["ema_b"].min()) < args.gate_thr
        return True

    if args.dry_inputs:   # decoded training rows: prior sites marked with ⟦ ⟧ (ctx: operands; text: numbers in the text)
        x, ops, ans = batch()
        for r in range(3):
            row = x[r].tolist()
            site = ops[ops[:, 0] == r][:, 1].tolist() if args.site == "ctx" else batch.tmeta[batch.tmeta[:, 0] == r][:, 1].tolist()
            show = sorted(set(range(max(0, len(row) - 40), len(row))) | {j for i in site for j in range(i - 6, i + 7) if 0 <= j < len(row)})
            txt, prev = "", None
            for j in show:
                txt += (" ... " if prev is not None and j != prev + 1 else "") + (f"⟦{tok.decode([row[j]])}⟧" if j in site else tok.decode([row[j]]))
                prev = j
            print(f"train row {r} (len {len(row)}, {len(site)} prior sites at {site[:12]}): {txt!r}")
        if args.site == "text":
            ns, ok = [], 0
            for _ in range(50):
                batch(); ns.append(len(batch.tmeta)); ok += text_ok()
            print(f"text numbers per batch over 50 batches: median {int(np.median(ns))}, min {min(ns)}; prior applied on {ok}/50 batches")
        return
    curve, t0 = [], time.time()
    model.train()
    ck = out / f"ckpt_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.pt"
    start = 0
    if args.ckpt_every and ck.exists():   # resume after preemption / requeue
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]; data_rng.bit_generator.state = st["data_rng"]
        dyn.update(st["dyn"]); curve = st["curve"]; start = st["step"] + 1
        torch.set_rng_state(st["torch_rng"].cpu())
        if dev == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"resumed from {ck.name} at step {start} of {steps}", flush=True)
    for s in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(s)
        x, ops, ans = batch()
        if s == 0:   # operand positions point at the operands' last tokens; answer spans decode to the sums
            for r_, p_, v_ in ops[:16].tolist():
                assert tok.decode([int(x[r_, p_])]) == tok.decode(enc(str(v_))[-1:]), (r_, p_, v_)
            for (r_, p_, n_, _, _), (_, _, a_), (_, _, b_) in zip(ans[:8].tolist(), ops[0:16:2].tolist(), ops[1:16:2].tolist()):
                assert tok.decode(x[r_, p_:p_ + n_].tolist()).strip() == str(a_ + b_), (r_, p_, n_, a_, b_)
            if args.site == "text":   # text positions decode to their values and lie before the arithmetic lines
                first_op = {int(r_): int(p_) for r_, p_, _ in ops[0::2 * args.arith_lines].tolist()}
                for r_, p_, v_ in batch.tmeta[:64].tolist():
                    assert tok.decode([int(x[r_, p_])]) == str(v_) and p_ < first_op[r_], (r_, p_, v_)
                show = [tok.decode(x[r_, max(0, p_ - 6):p_ + 3].tolist()) for r_, p_, _ in batch.tmeta[:6].tolist()]
                print(f"text site: {len(batch.tmeta)} numbers in the first batch, e.g. {show}", flush=True)
        on = prior_on(s)
        usable = args.site == "ctx" or text_ok()
        with torch.autocast(dev, dtype=torch.bfloat16):
            out_m = model(input_ids=x, labels=x)
            gl = site_prior(ops) if (on and usable) else torch.zeros((), device=dev)
            loss = out_m.loss + (args.lam * gl if (on and usable) else 0.0)
        with torch.no_grad():   # in-batch exact match per line (teacher forced): the gate's signal
            # answer AND the newline after it, as in the evals (a prefix of a longer prediction is not correct)
            A = torch.tensor(ans, device=dev)
            span = A[:, 2] + nl_len
            kk = torch.arange(int(span.max()), device=dev)
            pos = (A[:, 1:2] + kk).clamp(max=x.shape[1] - 1)
            valid = kk[None] < span[:, None]
            rr = A[:, 0:1].expand_as(pos)
            ok = ((out_m.logits[rr, pos - 1].argmax(-1) == x[rr, pos]) | ~valid).all(1)
            first = A[:, 3] == 1
            first_acc = float(ok[first].float().mean())
            for bk in range(len(BUCKETS)):   # per-bucket EMA over first lines (the evals' position)
                m = first & (A[:, 4] == bk)
                if bool(m.any()):
                    dyn["ema_b"][bk] = 0.98 * dyn["ema_b"][bk] + 0.02 * float(ok[m].float().mean())
        dyn["ema1"] = 0.98 * dyn["ema1"] + 0.02 * first_acc
        dyn["on"] += int(on)
        dyn["used"] = dyn.get("used", 0) + int(on and usable)
        dyn["toggles"] += int(dyn["last_on"] is not None and on != dyn["last_on"]); dyn["last_on"] = on
        grad_info = {}
        if (s % args.eval_every == 0 or s == steps - 1) and on and usable:   # LM vs prior pull, before clipping
            params = [q for q in model.parameters() if q.requires_grad]
            g1 = torch.autograd.grad(out_m.loss, params, retain_graph=True, allow_unused=True)
            g2 = torch.autograd.grad(args.lam * gl, params, retain_graph=True, allow_unused=True)
            v1 = torch.cat([(g if g is not None else torch.zeros_like(q)).flatten().float() for g, q in zip(g1, params)])
            v2 = torch.cat([(g if g is not None else torch.zeros_like(q)).flatten().float() for g, q in zip(g2, params)])
            grad_info = dict(grad_lm=float(v1.norm()), grad_prior=float(v2.norm()), grad_cos=float(v1 @ v2 / (v1.norm() * v2.norm() + 1e-12)))
            del g1, g2, v1, v2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % args.eval_every == 0 or s == steps - 1:
            if Ftab is not None and not (on and usable) and usable:   # prior off: still log the fit at its site
                with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                    gl = site_prior(ops)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):   # comparable across arms (incl. none)
                cka_ops_h3 = float(1 - prior(ops, Fref))
            r = evaluate(s); r.update(lm=float(out_m.loss), geo=float(gl) if usable else None, sec=time.time() - t0,
                                      prior_on=int(on), prior_used_frac=dyn["used"] / (s + 1), cka_ops_helix3=cka_ops_h3,
                                      text_numbers=int(len(batch.tmeta)) if args.site == "text" else None,
                                      prior_on_frac=dyn["on"] / (s + 1), ema_first_line=dyn["ema1"],
                                      ema_bucket_min=float(dyn["ema_b"].min()),
                                      ema_bucket={f"b{la}{lb}": float(v) for (la, lb), v in zip(BUCKETS, dyn["ema_b"])},
                                      prior_toggles=dyn["toggles"], **grad_info)
            curve.append(r)
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps({"args": vars(args), "steps": steps, "geo_layers": geo_layers, "curve": curve}, indent=1))
        if args.ckpt_every and (s + 1) % args.ckpt_every == 0 and s < steps - 1:
            tmp = ck.with_name(ck.name + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                        "data_rng": data_rng.bit_generator.state, "dyn": dict(dyn), "curve": curve, "step": s,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None}, tmp)
            tmp.replace(ck)
        if s == args.stop_after:
            print(f"stop_after {s}: exiting (test)", flush=True)
            return
    if ck.exists():
        ck.unlink()
    if args.save_model:
        torch.save(model.state_dict(), out / f"model_{args.tokenization}_{args.target}_seed{args.seed}{args.tag}.pt")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
