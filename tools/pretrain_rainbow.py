"""Rainbow step 2: several manifold priors during from-scratch PRETRAINING. FineWeb text with appended lines of three
synthetic tasks, each with its own geometry prior on its own tokens inside the training lines (the in-context site,
best for addition in pretraining: F69):
  add     "Q: a + b = c"            a, b in 10..99; unordered held-out split (as pretrain_helix --split unordered)
          prior "helix": the operand tokens a and b -> helix features of their values (periods 2, 5, 10, 100 + linear)
  days    "Q: Friday plus 3 days is Monday"     shifts k in 1..20, plus / minus; 30% of the (item, k, op) combos held out
          prior "days": the day token -> circle (k = 1) over the 7 days
  months  "Q: March minus 5 months is October"  as days, 12 months; prior "months": the month token -> circle
Arms (--priors): "none" or priors joined by "+", e.g. helix+days+months; "_shuf" = values permuted (fixed per seed);
suffix "@norm" divides lam by the number of priors. Each sequence ends with --lines lines, each of a random task.
Evaluation every --eval_every steps, in training position (held-out FineWeb chunk, then the prompt; the answer is one
token): held-out combos per task, trained combos per task, and validation LM loss.

    python tools/pretrain_rainbow.py --data <bins> --tokenizer <olmo2> --out <dir> --priors helix+days+months --seed 0
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
from pretrain_helix import build_model, centred_gram, cka

DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
TASKS = ("add", "days", "months")
PRIOR_TASK = {"helix": "add", "days": "days", "months": "months"}


def unordered_split():
    U = np.array([(a, b) for a in range(10, 100) for b in range(a, 100)])
    U = U[np.random.default_rng(0).permutation(len(U))]
    held = U[:500]
    test = np.array([(a, b) for a, b in held.tolist()] + [(b, a) for a, b in held.tolist() if a != b])
    hs = set(map(tuple, test.tolist()))
    trainp = np.array([(a, b) for a in range(10, 100) for b in range(10, 100) if (a, b) not in hs])
    return test, trainp


def cyclic_split(items, kmax=20, frac=0.3, seed=0):
    combos = [(i, k, op) for i in range(len(items)) for k in range(1, kmax + 1) for op in ("plus", "minus")]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combos))
    nt = int(frac * len(combos))
    test = [combos[j] for j in order[:nt]]
    train = [combos[j] for j in order[nt:]]
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--priors", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=float, default=3e8)
    ap.add_argument("--lines", type=int, default=6, help="task lines appended to every sequence (task drawn uniformly)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--geo_layers", default=None)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--dry_inputs", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    norm = args.priors.endswith("@norm")
    active = [] if args.priors == "none" else args.priors.removesuffix("@norm").split("+")
    assert all(p.removesuffix("_shuf") in PRIOR_TASK for p in active), active
    lam = args.lam / max(1, len(active)) if norm else args.lam
    path = out / f"run_{args.priors}_seed{args.seed}.json"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint32, mode="r")
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint32, mode="r")
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    # ---------------------------------------------------------------- task lines (+ positions of the shaped tokens)
    test_add, train_add = unordered_split()
    cyc = {"days": (DAYS, *cyclic_split(DAYS, seed=1)), "months": (MONTHS, *cyclic_split(MONTHS, seed=2))}

    def add_line(a, b):
        ids = enc(f"Q: {a} + {b} = {a + b}\n")
        assert len(ids) == 11 and tok.decode([ids[3]]) == str(a) and tok.decode([ids[6]]) == str(b) and tok.decode([ids[9]]) == str(a + b)
        return ids, [(3, a), (6, b)], 9          # operand positions and values; answer index

    def cyc_line(task, i, k, op):
        items = cyc[task][0]
        j = (i + k) % len(items) if op == "plus" else (i - k) % len(items)
        ids = enc(f"Q: {items[i]} {op} {k} {task} is {items[j]}\n")
        assert tok.decode([ids[2]]).strip() == items[i] and tok.decode([ids[-2]]).strip() == items[j], ids
        return ids, [(2, i)], len(ids) - 2

    pools = {"add": [("add", int(a), int(b)) for a, b in train_add.tolist()]}
    for t in ("days", "months"):
        pools[t] = [(t, *c) for c in cyc[t][1]]
    def make(spec):
        return add_line(spec[1], spec[2]) if spec[0] == "add" else cyc_line(*spec)
    cache = {}
    def line_of(spec):
        if spec not in cache:
            cache[spec] = make(spec)
        return cache[spec]

    # ---------------------------------------------------------------- eval sets (training position)
    vr = np.random.default_rng(11)
    def evalset(specs):
        rows, ans = [], []
        for sp in specs:
            ids, _, ai = line_of(sp) if sp in cache else make(sp)
            prompt = ids[:ai]
            n = args.seq - 2 - len(prompt)   # the answer would sit at seq - 2, as in training
            o = int(vr.integers(0, len(val) - n))
            rows.append(np.concatenate([np.asarray(val[o:o + n], dtype=np.int64), prompt]))
            ans.append(ids[ai])
        return torch.tensor(np.stack(rows), device=dev), torch.tensor(ans, device=dev)
    er = np.random.default_rng(5)
    evalsets = {"add_held": evalset([("add", int(a), int(b)) for a, b in test_add[er.permutation(len(test_add))[:600]].tolist()]),
                "add_train": evalset([pools["add"][j] for j in er.permutation(len(pools["add"]))[:300]])}
    for t in ("days", "months"):
        evalsets[f"{t}_held"] = evalset([(t, *c) for c in cyc[t][2]])
        evalsets[f"{t}_train"] = evalset([pools[t][j] for j in er.permutation(len(pools[t]))[:150]])

    # ---------------------------------------------------------------- targets (value -> features), shuffled per seed
    rs = np.random.default_rng(124 + args.seed)
    tab = {"helix": nl.features(np.arange(100))[0],
           "days": np.stack([np.cos(2 * np.pi * np.arange(7) / 7), np.sin(2 * np.pi * np.arange(7) / 7)], 1),
           "months": np.stack([np.cos(2 * np.pi * np.arange(12) / 12), np.sin(2 * np.pi * np.arange(12) / 12)], 1)}
    pm = np.arange(100); pm[10:] = 10 + rs.permutation(90)
    tab["helix_shuf"] = tab["helix"][pm]
    tab["days_shuf"] = tab["days"][rs.permutation(7)]
    tab["months_shuf"] = tab["months"][rs.permutation(12)]
    if args.dry_inputs:
        for sp in (pools["add"][0], pools["days"][0], pools["months"][0]):
            ids, pos, ai = line_of(sp)
            print(sp, [tok.decode([t]) for t in ids], "shaped", [(p, tok.decode([ids[p]])) for p, _ in pos], "answer", tok.decode([ids[ai]]))
        for k, (x, a) in evalsets.items():
            print(f"{k}: n {len(x)}  ...{tok.decode(x[0, -12:].tolist())!r} -> {tok.decode([int(a[0])])!r}")
        print("active", active, "lam", lam)
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
    rng = np.random.default_rng(1000 + args.seed)

    def batch():
        rows, shaped = [], {t: [] for t in TASKS}
        for r in range(args.bs):
            specs = []
            for _ in range(args.lines):
                t = TASKS[int(rng.integers(3))]
                specs.append(pools[t][int(rng.integers(len(pools[t])))])
            parts = [line_of(sp) for sp in specs]
            n = args.seq - sum(len(p[0]) for p in parts)
            o = int(rng.integers(0, len(train) - n))
            row = list(np.asarray(train[o:o + n], dtype=np.int64))
            off = n
            for sp, (ids, pos, _) in zip(specs, parts):
                for p, v in pos:
                    shaped[sp[0]].append((r, off + p, v))
                row += ids; off += len(ids)
            rows.append(row)
        return torch.tensor(np.array(rows), device=dev), {t: np.array(v) for t, v in shaped.items()}

    def prior_loss(name, sh):
        m = sh[PRIOR_TASK[name.removesuffix("_shuf")]]
        if len(m) < 4:
            return torch.zeros((), device=dev)
        r, pos = torch.tensor(m[:, 0], device=dev), torch.tensor(m[:, 1], device=dev)
        K = centred_gram(tab[name][m[:, 2]], dev)
        return sum(1 - cka(store[L][r, pos], K) for L in geo_layers) / len(geo_layers)

    @torch.no_grad()
    def evaluate(step) -> dict:
        model.eval()
        res = {"step": step, "tokens": step * args.bs * args.seq}
        with torch.autocast(dev, dtype=torch.bfloat16):
            for name, (ids, ans) in evalsets.items():
                hit = [model(input_ids=ids[i:i + 250], logits_to_keep=1).logits[:, -1].argmax(-1) == ans[i:i + 250] for i in range(0, len(ids), 250)]
                res[name] = float(torch.cat(hit).float().mean())
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
        x, sh = batch()
        if s == 0:   # shaped positions hold the intended tokens
            for t in TASKS:
                for r_, p_, v_ in sh[t][:8].tolist():
                    want = str(v_) if t == "add" else (DAYS if t == "days" else MONTHS)[v_]
                    assert tok.decode([int(x[r_, p_])]).strip() == want, (t, r_, p_, v_)
        with torch.autocast(dev, dtype=torch.bfloat16):
            out_m = model(input_ids=x, labels=x)
            geo = {p: prior_loss(p, sh) for p in active}
            loss = out_m.loss + lam * sum(geo.values()) if geo else out_m.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % args.eval_every == 0 or s == steps - 1:
            r = evaluate(s); r.update(lm=float(out_m.loss), sec=time.time() - t0, **{f"geo_{p}": float(v) for p, v in geo.items()})
            curve.append(r)
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps({"args": vars(args), "lam_effective": lam, "steps": steps, "geo_layers": geo_layers, "curve": curve}, indent=1))
    if args.save_model:
        torch.save(model.state_dict(), out / f"model_{args.priors}_seed{args.seed}.pt")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
