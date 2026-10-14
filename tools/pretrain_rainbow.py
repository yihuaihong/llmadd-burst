"""Rainbow step 2: several manifold priors during from-scratch PRETRAINING. FineWeb text with appended lines of three
synthetic tasks, each with its own geometry prior on its own tokens inside the training lines (the in-context site,
best for addition in pretraining: F69):
  add     "Q: a + b = c"            a, b in 10..99; unordered held-out split (as pretrain_helix --split unordered)
          prior "helix": the operand tokens a and b -> helix features of their values (periods 2, 5, 10, 100 + linear)
  days    "Q: Friday plus 3 days is Monday"     shifts k in 1..20, plus / minus. Held out by EQUIVALENCE CLASS
          (item, effective shift e = +-k mod 7): every combo of a held (item, e) is never trained, so a held-out
          problem needs the shift learned on OTHER items (audit 2026-10-13: a per-combo split left an equivalent
          trained combo, e.g. Fri + 2 for Fri + 9, for 100% of the held-out days problems); e = 0 is not held out
          prior "days": the day token -> circle (k = 1) over the 7 days
  months  "Q: March minus 5 months is October"  as days, 12 months; prior "months": the month token -> circle
Arms (--priors): "none" or priors joined by "+", e.g. helix+days+months; "_shuf" = values permuted (per seed, re-drawn
until CKA(shuffled, true) < 0.15); suffix "@norm" divides lam by the number of priors; suffix "@bal" rescales each
prior's lam at step 0 so that every prior's gradient norm equals the smallest one's (the days / months priors pull
3-4x harder than the helix at the same lam). Each sequence ends with --lines lines, each of a random task.
Logged at every eval, for EVERY arm: the gradient norm of the LM loss and of each of the three true priors (active or
not) and their cosines on the current batch, and the CKA of the item / operand states in the eval prompts to all
three true geometries; held-out accuracy split by k <= 9 / k >= 10 (the numbers 10..20 are the add operand tokens the
helix shapes) and add by sum < 100 / >= 100 (tied embeddings: those answers share rows with operand tokens).
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
    """Hold out (item, effective shift) classes, e != 0; all (k, op) combos of a held class go to test."""
    n = len(items)
    eff = lambda k, op: (k if op == "plus" else -k) % n
    classes = [(i, e) for i in range(n) for e in range(1, n)]
    rng = np.random.default_rng(seed)
    held = {classes[j] for j in rng.permutation(len(classes))[:int(round(frac * len(classes)))]}
    combos = [(i, k, op) for i in range(n) for k in range(1, kmax + 1) for op in ("plus", "minus")]
    test = [c for c in combos if (c[0], eff(c[1], c[2])) in held]
    train = [c for c in combos if (c[0], eff(c[1], c[2])) not in held]
    assert not ({(i, eff(k, op)) for i, k, op in test} & {(i, eff(k, op)) for i, k, op in train})
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
    norm, bal = args.priors.endswith("@norm"), args.priors.endswith("@bal")
    active = [] if args.priors == "none" else args.priors.removesuffix("@norm").removesuffix("@bal").split("+")
    assert all(p.removesuffix("_shuf") in PRIOR_TASK for p in active), active
    lam = args.lam / max(1, len(active)) if norm else args.lam
    lam_p = {p: lam for p in active}   # per prior; @bal rescales at step 0
    import hashlib
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path = out / f"run_{args.priors}_seed{args.seed}.json"
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
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    # ---------------------------------------------------------------- task lines (+ positions of the shaped tokens)
    test_add, train_add = unordered_split()
    cyc = {"days": (DAYS, *cyclic_split(DAYS, seed=10 + args.seed)), "months": (MONTHS, *cyclic_split(MONTHS, seed=20 + args.seed))}

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
    shaped_eval = {}
    def evalset(specs, name=None):
        rows, ans, sh = [], [], []
        for r, sp in enumerate(specs):
            ids, pos, ai = line_of(sp) if sp in cache else make(sp)
            prompt = ids[:ai]
            n = args.seq - 2 - len(prompt)   # the answer would sit at seq - 2, as in training
            o = int(vr.integers(0, len(val) - n))
            rows.append(np.concatenate([np.asarray(val[o:o + n], dtype=np.int64), prompt]))
            ans.append(ids[ai])
            sh += [(r, n + p, v) for p, v in pos]
        if name:
            shaped_eval[name] = np.array(sh)
        return torch.tensor(np.stack(rows), device=dev), torch.tensor(ans, device=dev)
    er = np.random.default_rng(5)
    add_held_specs = [("add", int(a), int(b)) for a, b in test_add[er.permutation(len(test_add))[:600]].tolist()]
    evalsets = {"add_held": evalset(add_held_specs, "add_held"),
                "add_train": evalset([pools["add"][j] for j in er.permutation(len(pools["add"]))[:300]])}
    sub_masks = {"add_held": {"lo": np.array([a + b < 100 for _, a, b in add_held_specs])}}
    for t in ("days", "months"):
        held_specs = [(t, *c) for c in cyc[t][2]]
        evalsets[f"{t}_held"] = evalset(held_specs, f"{t}_held")
        evalsets[f"{t}_train"] = evalset([pools[t][j] for j in er.permutation(len(pools[t]))[:150]])
        sub_masks[f"{t}_held"] = {"k1to9": np.array([sp[2] <= 9 for sp in held_specs])}

    # ---------------------------------------------------------------- targets (value -> features), shuffled per seed
    rs = np.random.default_rng(124 + args.seed)
    tab = {"helix": nl.features(np.arange(100))[0],
           "days": np.stack([np.cos(2 * np.pi * np.arange(7) / 7), np.sin(2 * np.pi * np.arange(7) / 7)], 1),
           "months": np.stack([np.cos(2 * np.pi * np.arange(12) / 12), np.sin(2 * np.pi * np.arange(12) / 12)], 1)}
    def gram_np(F):
        F = (F - F.mean(0)) / (F.std(0) + 1e-9); G = F @ F.T; return G - G.mean(0) - G.mean(1)[:, None] + G.mean()
    def wrong(F, lo, n):   # permutation of rows lo..lo+n-1 with CKA(shuffled, true) < 0.15
        Gt = gram_np(F[lo:lo + n])
        for _ in range(5000):
            q = rs.permutation(n); Gs = gram_np(F[lo:lo + n][q])
            if (Gs * Gt).sum() / (np.linalg.norm(Gs) * np.linalg.norm(Gt)) < 0.15:
                out = F.copy(); out[lo:lo + n] = F[lo:lo + n][q]; return out
        raise RuntimeError("no far permutation")
    tab["helix_shuf"] = wrong(tab["helix"], 10, 90)
    tab["days_shuf"] = wrong(tab["days"], 0, 7)
    tab["months_shuf"] = wrong(tab["months"], 0, 12)
    if args.dry_inputs:
        for sp in (pools["add"][0], pools["days"][0], pools["months"][0]):
            ids, pos, ai = line_of(sp)
            print(sp, [tok.decode([t]) for t in ids], "shaped", [(p, tok.decode([ids[p]])) for p, _ in pos], "answer", tok.decode([ids[ai]]))
        for k, (x, a) in evalsets.items():
            print(f"{k}: n {len(x)}  ...{tok.decode(x[0, -12:].tolist())!r} -> {tok.decode([int(a[0])])!r}")
        print("active", active, "lam", lam, "code", code_sha)
        print("held classes: days", len({(i, (k if op == 'plus' else -k) % 7) for i, k, op in cyc['days'][2]}),
              "months", len({(i, (k if op == 'plus' else -k) % 12) for i, k, op in cyc['months'][2]}))
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
                hit, states = [], {L: [] for L in geo_layers}
                for i in range(0, len(ids), 250):
                    hit.append(model(input_ids=ids[i:i + 250], logits_to_keep=1).logits[:, -1].argmax(-1) == ans[i:i + 250])
                    if name in shaped_eval:
                        m = shaped_eval[name]; m = m[(m[:, 0] >= i) & (m[:, 0] < i + 250)]
                        for L in geo_layers:
                            states[L].append((store[L][torch.tensor(m[:, 0] - i, device=dev), torch.tensor(m[:, 1], device=dev)].float(), m[:, 2]))
                h = torch.cat(hit).float()
                res[name] = float(h.mean())
                for sub, mask in sub_masks.get(name, {}).items():
                    mk = torch.tensor(mask, device=dev)
                    res[f"{name}_{sub}"] = float(h[mk].mean()); res[f"{name}_not{sub}"] = float(h[~mk].mean())
                if name in shaped_eval:   # geometry of the shaped tokens in held-out prompts, against the TRUE targets
                    geo_name = {"add_held": "helix", "days_held": "days", "months_held": "months"}[name]
                    for L in geo_layers:
                        H = torch.cat([a for a, _ in states[L]]); v = np.concatenate([b for _, b in states[L]])
                        res[f"cka_{geo_name}@L{L}"] = float(cka(H, centred_gram(tab[geo_name][v], dev)))
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
        grad_info = {}
        if s == 0 or s % args.eval_every == 0 or s == steps - 1:   # every arm: LM and all three TRUE priors
            with torch.autocast(dev, dtype=torch.bfloat16):
                terms = {"lm": out_m.loss, **{p: (geo[p] if p in geo else prior_loss(p, sh)) for p in ("helix", "days", "months")}}
            params = [q for q in model.parameters() if q.requires_grad]
            vec = {}
            for k_, v_ in terms.items():
                g = torch.autograd.grad(v_, params, retain_graph=True, allow_unused=True)
                vec[k_] = torch.cat([(a if a is not None else torch.zeros_like(q)).flatten().float() for a, q in zip(g, params)])
            names = list(vec)
            grad_info = {f"gnorm_{k_}": float(vec[k_].norm()) for k_ in names}
            grad_info.update({f"gcos_{a}|{b}": float(vec[a] @ vec[b] / (vec[a].norm() * vec[b].norm() + 1e-12))
                              for i_, a in enumerate(names) for b in names[i_ + 1:]})
            if s == 0 and bal and len(active) > 1:   # equalise the priors' pull: each lam_p * |grad p| = the smallest
                gn = {}
                for p_ in active:
                    g = torch.autograd.grad(geo[p_], params, retain_graph=True, allow_unused=True)
                    gn[p_] = float(torch.cat([(a if a is not None else torch.zeros_like(q)).flatten().float() for a, q in zip(g, params)]).norm())
                ref = min(gn.values())
                lam_p = {p_: lam * ref / gn[p_] for p_ in active}
                print("balanced lambdas", lam_p, "from gradient norms", gn, flush=True)
            del vec
        loss = out_m.loss + sum(lam_p[p] * v for p, v in geo.items()) if geo else out_m.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % args.eval_every == 0 or s == steps - 1:
            r = evaluate(s); r.update(lm=float(out_m.loss), sec=time.time() - t0, **{f"geo_{p}": float(v) for p, v in geo.items()}, **grad_info)
            curve.append(r)
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps({"args": vars(args), "lam_effective": lam_p, "code_sha256": code_sha, "steps": steps,
                                        "geo_layers": geo_layers, "curve": curve}, indent=1))
    if args.save_model:
        torch.save(model.state_dict(), out / f"model_{args.priors}_seed{args.seed}.pt")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
