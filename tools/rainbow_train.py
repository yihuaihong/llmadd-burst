"""Multi-prior ("rainbow") fine-tuning: several structured tasks trained TOGETHER, each with its own geometry prior that
can be switched on or off independently. Question: do priors on (near-)orthogonal concept manifolds add up, or
interfere? (Rainbow-style: single priors, all priors, all-but-one, all shuffled.)

Tasks (same data and splits as the single-domain experiments; but multi-task training, the shared optimiser and the
step counts differ, so only paired comparisons within this script are valid, not with F14 / F24 numbers):
  add     "Q: a + b = " -> a + b; pairs a, b in 10..99 permuted by rng(0): train pairs[:1500], test pairs[1500:2500]
          (manifold_train.py defaults); test_strict = test pairs whose swap (b, a) is not in train
  days    concept_train.build_data(days): "Q: Friday plus 3 days is" -> " Monday", (item, k, op) combos, k 1..40,
          60 / 40 train / test; test_k1to9 / test_k10plus split the test set by the shift k (k >= 10 is a two-digit
          number token, directly shaped by the helix prior: F44 E4a showed helix alone helps months / days);
          k41to80 = larger shifts (called "ood" in concept_train, but here these numbers are trained as add operands)
  months  the same for months
Priors (each adds lam * (1 - CKA) between the states of "Q: <item>" at --layers and the target Gram):
  helix   numbers 10..99, helix features (periods 2, 5, 10, 100 + linear)          -> shapes the add manifold
  days    circle (k = 1) over the 7 days                                            -> shapes the days manifold
  months  circle over the 12 months                                                 -> shapes the months manifold
  <name>_shuf  the same target with the items permuted (fixed per seed): wrong geometry, same strength
Arms (--arms, comma list): "none" or priors joined by "+", e.g. none,helix,days,months,helix+days+months,days+months;
a suffix "@norm" divides lambda by the number of active priors (same total constraint strength as one prior).
Shuffled targets are re-drawn until CKA(true, shuffled) < 0.15, so every seed's shuffle is clearly wrong.
At step 0 of every run: gradient norm of each loss term on the LoRA parameters and their pairwise cosines.
Each step: one batch of --bs from EVERY task; loss = sum of the task CEs (each task gets the gradient it gets in a
single-task run) + the active priors. LoRA (rank --rank) on all linear layers, as in the single-domain runs.

    python tools/rainbow_train.py --model <ckpt> --out <dir> --arms none,helix,days,months,helix+days+months --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import numlib as nl
from concept_train import DOMAINS, build_data, centred_gram, items_and_values, target_features
from manifold_train import cka, decoder_layers

TASKS = ("add", "days", "months")
PRIORS = ("helix", "days", "months")


def add_data(tok):
    rng = np.random.default_rng(0)
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    pairs = pairs[rng.permutation(len(pairs))]
    train, test = pairs[:1500], pairs[1500:2500]
    tr = set(map(tuple, train.tolist()))
    assert not (set(map(tuple, test.tolist())) & tr)
    strict = np.array([(b, a) not in tr for a, b in test.tolist()])
    mk = lambda P: [(f"Q: {a} + {b} = ", str(a + b)) for a, b in P]
    return {"train": mk(train), "test": mk(test), "test_strict": mk(test[strict])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="none,helix,days,months,helix+days+months")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test: a small random OLMo-2 with the given tokenizer")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    for a in arms:
        assert a == "none" or all(p.removesuffix("_shuf") in PRIORS for p in a.removesuffix("@norm").split("+")), a
    seeds = [int(s) for s in args.seeds.split(",")]
    layers_geo = [int(v) for v in args.layers.split(",")]
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.tiny:
        from transformers import Olmo2Config, Olmo2ForCausalLM
        dev = "cpu"
        model = Olmo2ForCausalLM(Olmo2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                             num_attention_heads=4, num_key_value_heads=4))
        layers_geo = [1, 2, 3]   # decoder_layers() is 0-based: indices 0..3 on the tiny model
    else:
        dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    enc = lambda ps: nl.encode(tok, ps, dev)

    # ---------------------------------------------------------------- data (+ intent asserts)
    data = {"add": add_data(tok)}
    items = {}
    for d in ("days", "months"):
        it, vals = items_and_values(DOMAINS[d])
        items[d] = (it, vals)
        data[d] = build_data(DOMAINS[d], it, vals, np.random.default_rng(0))   # identical to concept_train's split
        data[d]["k41to80"] = data[d].pop("ood")
        data[d]["test_k1to9"] = [q for q in data[d]["test"] if int(q[0].split()[3]) <= 9]
        data[d]["test_k10plus"] = [q for q in data[d]["test"] if int(q[0].split()[3]) >= 10]
    evalsets = {}
    for t, splits in data.items():
        for name, probs in splits.items():
            ids = enc([p for p, _ in probs])
            ans = [tok(a, add_special_tokens=False)["input_ids"] for _, a in probs]
            assert all(len(a) == 1 for a in ans), (t, name, "answers must be single tokens")
            evalsets[(t, name)] = (ids, torch.tensor([a[0] for a in ans], device=dev))
    prefix = {"helix": enc([f"Q: {x}" for x in range(10, 100)]),
              "days": enc([f"Q: {w}" for w in items["days"][0]]), "months": enc([f"Q: {w}" for w in items["months"][0]])}
    # the state is read at the LAST position of each prefix, which must be the item itself (one token)
    names = {"helix": [str(x) for x in range(10, 100)], "days": items["days"][0], "months": items["months"][0]}
    for k, P in prefix.items():
        assert [tok.decode([int(t)]).strip() for t in P[:, -1]] == names[k], k
    for t in ("days", "months"):   # the item is the 3rd token of every task prompt, as in the prefix
        ids = evalsets[(t, "train")][0]
        assert set(ids[:, 2].tolist()) <= set(prefix[t][:, -1].tolist()), t
    def targets(seed):
        rs = np.random.default_rng(124 + seed)
        def wrong(make, n):   # re-draw the permutation until the shuffled target is far from the true one
            true = centred_gram(make(np.arange(n)), "cpu")
            for _ in range(1000):
                pm = rs.permutation(n)
                F = make(pm)
                G = centred_gram(F, "cpu")
                if float((G * true).sum() / (G.norm() * true.norm())) < 0.15:
                    return F
            raise RuntimeError("no permutation far enough from the true target")
        xs = np.arange(10, 100)
        T = {"helix": nl.features(xs)[0], "helix_shuf": wrong(lambda pm: nl.features(xs[pm])[0], 90)}
        for d in ("days", "months"):
            dom, (it, vals) = DOMAINS[d], items[d]
            T[d] = target_features(dom, "circle", vals, None)
            T[d + "_shuf"] = wrong(lambda pm, dom=dom, vals=vals: target_features(dom, "circle_shuf", vals, pm), len(vals))
        return {k: centred_gram(v, dev) for k, v in T.items()}
    print("inputs (decoded):")
    for t in TASKS:
        ids, ans = evalsets[(t, "train")]
        for r in range(2):
            print(f"  {t:6s} {tok.decode(ids[r].tolist())!r} -> {tok.decode([int(ans[r])])!r}")
    print("  prefixes:", {k: tok.decode(v[0].tolist()) for k, v in prefix.items()})
    print("  n:", {f"{t}/{n}": len(v[0]) for (t, n), v in evalsets.items()})

    # ---------------------------------------------------------------- model
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    layers = decoder_layers(model)
    store = {}
    for L in layers_geo:
        layers[L].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))
    autocast = torch.autocast(dev, dtype=torch.bfloat16, enabled=False)   # as manifold_train / concept_train in LoRA mode

    def forward(ids):
        with autocast:
            return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()

    def prior_loss(name, K):
        forward(prefix[name.removesuffix("_shuf")])
        return sum(1 - cka(store[L][:, -1, :], K) for L in layers_geo) / len(layers_geo)

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        res = {}
        for (t, name), (ids, ans) in evalsets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs]).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[f"{t}/{name}"] = float((am == ans).float().mean())
        return res

    @torch.no_grad()
    def geometry(K) -> dict:
        model.eval()
        rep = {}
        for p in PRIORS:
            forward(prefix[p])
            rep[p] = {f"L{L}": float(cka(store[L][:, -1, :], K[p])) for L in layers_geo}
        return rep

    def reset(seed):
        torch.manual_seed(seed)
        from peft.tuners.lora import LoraLayer
        for m in model.modules():
            if isinstance(m, LoraLayer):
                m.reset_lora_parameters("default", True)

    K0 = targets(0)
    base = {"eval": evaluate(), "geometry": geometry(K0), "n": {f"{t}/{n}": len(v[0]) for (t, n), v in evalsets.items()}}
    (out / "base.json").write_text(json.dumps(base, indent=1))
    print(f"base {base['eval']} ({time.time() - t0:.0f}s)", flush=True)
    for seed in seeds:
        K = targets(seed)
        for arm in arms:
            path = out / f"run_{arm}_seed{seed}.json"
            if path.exists():
                continue
            active = [] if arm == "none" else arm.removesuffix("@norm").split("+")
            lam = args.lam / len(active) if arm.endswith("@norm") else args.lam
            reset(seed)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
            rng = np.random.default_rng(1000 + seed)
            orders = {t: [] for t in TASKS}
            log = []
            model.train()
            params = [p for p in model.parameters() if p.requires_grad]
            grad_diag = None
            for s in range(args.steps):
                opt.zero_grad(set_to_none=True)
                ce = {}
                for t in TASKS:
                    ids, ans = evalsets[(t, "train")]
                    if len(orders[t]) < args.bs:
                        orders[t] = list(rng.permutation(len(ids)))
                    bi = torch.tensor([orders[t].pop() for _ in range(args.bs)], device=dev)
                    ce[t] = Fn.cross_entropy(forward(ids[bi]), ans[bi])
                geo = {p: prior_loss(p, K[p]) for p in active}
                if s == 0:   # which terms pull the shared LoRA parameters in which direction (all three true priors)
                    terms = {f"ce_{t}": v for t, v in ce.items()}
                    terms.update({f"prior_{p}": (geo[p] if p in geo else prior_loss(p, K[p])) for p in PRIORS})
                    g = {}
                    for name, v in terms.items():
                        gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                        g[name] = torch.cat([(x if x is not None else torch.zeros_like(q)).flatten().float() for x, q in zip(gs, params)])
                    names = list(g)
                    grad_diag = {"norm": {n: float(g[n].norm()) for n in names},
                                 "cos": {f"{a}|{b}": float(g[a] @ g[b] / (g[a].norm() * g[b].norm() + 1e-12))
                                         for i, a in enumerate(names) for b in names[i + 1:]}}
                    del g
                loss = sum(ce.values()) + lam * sum(geo.values())
                loss.backward()
                opt.step()
                if s % 40 == 0 or s == args.steps - 1:
                    log.append({"step": s, **{f"ce_{t}": float(v) for t, v in ce.items()}, **{f"geo_{p}": float(v) for p, v in geo.items()}})
            del opt
            ev = evaluate()
            path.write_text(json.dumps({"arm": arm, "active": active, "seed": seed, "lam": lam, "grad_diag": grad_diag, "rank": args.rank,
                                        "lr": args.lr, "steps": args.steps, "eval": ev, "geometry": geometry(K), "log": log}, indent=1))
            print(f"{arm} seed{seed}: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items() if k.endswith("test") or k.endswith("strict"))
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    summarize(out)


def summarize(out: Path) -> None:
    from scipy import stats
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    if not runs:
        return
    keys = [k for k in runs[0]["eval"] if k.endswith(("/test", "/test_strict", "/test_k1to9", "/test_k10plus", "/k41to80"))]
    none = {r["seed"]: r for r in runs if r["arm"] == "none"}
    lines = ["| arm | n | " + " | ".join(keys) + " |\n", "|---|---|" + "---|" * len(keys) + "\n"]
    for arm in dict.fromkeys(r["arm"] for r in runs):
        rs = [r for r in runs if r["arm"] == arm]
        cells = []
        for k in keys:
            v = np.array([r["eval"][k] for r in rs])
            if arm == "none" or not none:
                cells.append(f"{100 * v.mean():.1f}")
            else:
                pair = [(r["eval"][k], none[r["seed"]]["eval"][k]) for r in rs if r["seed"] in none]
                d = np.array([a - b for a, b in pair])
                p = stats.ttest_rel(*zip(*pair)).pvalue if len(pair) > 1 else float("nan")
                cells.append(f"{100 * v.mean():.1f} ({100 * d.mean():+.1f}, p {p:.2f})")
        lines.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + " |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
