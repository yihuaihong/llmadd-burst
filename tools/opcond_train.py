"""Operation-conditioned geometry priors (fine-tuning): can multiplication benefit if its operands are shaped in the
MULTIPLICATION context only, instead of one shared number geometry for every operation?

Background: one shared helix on "Q: x" helps add / sub but hurts mul (F52: 42B +13 / +19 / -12), and every shared
geometry tried for mul hurt it (log, factor, digit, FoNE: F53, F57). Here each operation can get its own prior on the
operand states INSIDE its own prompts ("Q: a * b = ": the states of the tokens a and b, from the training forward
itself), so the add context can be a helix while the mul context is logarithmic.
ONLY the second operand b is shaped: with causal attention the state of a (position 3) has not seen the operator
(position 4), so it is identical in add / sub / mul prompts and equal to the shared "Q: a" state (audit 2026-10-13).

Tasks (identical to manifold_train.py --task mixed): add 1500 / 1000, sub 1500 / 1000, mul (a * b <= 999) 900 / 500
train / test pairs, a, b in 10..99, permuted by one rng(0) in the order add, sub, mul; prompt "Q: a <op> b = ".
Priors (comma-free names joined by "+" in an arm):
  prefix_helix        helix on the last-token states of "Q: x", x = 10..99, at --layers (the original shared prior)
  <op>_<geom>         CKA on the state of the operand b (position 6, after the operator) in prompts of operation <op>
                      (add / sub / mul), on a separate draw of --prior_bs training prompts of that operation every
                      step; target = manifold_train.geom_features(<geom>) of b, standardised over all values 10..99
                      (not per batch) and centred in the batch; e.g. add_helix, sub_helix, mul_log, mul_factor;
                      "<op>_<geom>_shuf" permutes the values 10..99 (fixed per seed)
A suffix "@norm" on an arm divides lambda by the number of active priors. Step 0 of every run logs the gradient norm
of every CE and prior term on the LoRA parameters and their pairwise cosines.
Main metrics: mul/test_strict (58% of the mul test pairs have their swap trained: mul/test is mostly recall), and
mul/train next to it. Only paired comparisons within this script are valid: training differs from F52 (per-operation
batches, summed CEs, 280 steps; mul sees ~5 epochs).
Each step: one batch of --bs per operation; loss = sum of the CEs + lam * sum of the active priors.
Eval: test accuracy per operation, and test_strict (pairs whose swap is not trained) for add and mul.

    python tools/opcond_train.py --model <ckpt> --out <dir> --arms none,prefix_helix,add_helix+sub_helix+mul_log --seeds 0,1,2
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
from manifold_train import OPS, answers, cka, decoder_layers, geom_features
from concept_train import centred_gram

TASKS = ("add", "sub", "mul")


def mixed_data():
    mrng = np.random.default_rng(0)
    out = {}
    for o, ntr, nte in (("add", 1500, 1000), ("sub", 1500, 1000), ("mul", 900, 500)):
        P = np.array([(a, b) for a in range(10, 100) for b in range(10, 100) if (o != "sub" or a >= b) and (o != "mul" or a * b <= 999)])
        P = P[mrng.permutation(len(P))]
        out[o] = (P[:ntr], P[ntr:ntr + nte])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="none,prefix_helix,add_helix+sub_helix,add_helix+sub_helix+mul_log,mul_log")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--prior_bs", type=int, default=64, help="prompts of an operation drawn for its in-context prior each step")
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test: a small random OLMo-2 with the given tokenizer")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    for a in arms:
        for p in ([] if a == "none" else a.removesuffix("@norm").split("+")):
            assert p == "prefix_helix" or (p.split("_")[0] in TASKS and len(p.split("_")) >= 2), p
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
        layers_geo = [1, 2, 3]
    else:
        dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    enc = lambda ps: nl.encode(tok, ps, dev)

    # ---------------------------------------------------------------- data (+ intent asserts)
    data = mixed_data()
    num_id = {v: tok(str(v), add_special_tokens=False)["input_ids"] for v in range(0, 1000)}
    assert all(len(t) == 1 for t in num_id.values()), "numbers 0..999 must be single tokens"
    num_id = {v: t[0] for v, t in num_id.items()}
    sets = {}
    for o in TASKS:
        tr, te = data[o]
        trs = set(map(tuple, tr.tolist()))
        assert not (set(map(tuple, te.tolist())) & trs)
        named = {"train": tr, "test": te}
        if o in ("add", "mul"):
            named["test_strict"] = te[np.array([(b, a) not in trs for a, b in te.tolist()])]
        for name, P in named.items():
            ids = enc([f"Q: {a} {OPS[o]} {b} = " for a, b in P])
            ans = torch.tensor([num_id[int(v)] for v in answers(o, P)], device=dev)
            sets[(o, name)] = (ids, ans, P)
        ids, _, P = sets[(o, "train")]
        # operand tokens sit at positions 3 and 6 of "Q: a <op> b = " (tokens Q, :, ' ', a, ' op', ' ', b, ' =', ' ')
        assert all(int(ids[i, 3]) == num_id[a] and int(ids[i, 6]) == num_id[b] for i, (a, b) in enumerate(P.tolist())), o
    prefix = enc([f"Q: {x}" for x in range(10, 100)])
    assert [tok.decode([int(t)]).strip() for t in prefix[:, -1]] == [str(x) for x in range(10, 100)]
    print("inputs (decoded):")
    for o in TASKS:
        ids, ans, _ = sets[(o, "train")]
        print(f"  {o}: {tok.decode(ids[0].tolist())!r} -> {tok.decode([int(ans[0])])!r}; operands at 3, 6: "
              f"{tok.decode([int(ids[0, 3])])!r} {tok.decode([int(ids[0, 6])])!r}")
    print("  n:", {f"{o}/{n}": len(v[0]) for (o, n), v in sets.items()})

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

    def forward(ids):
        return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()

    def perm_for(seed):
        pm = np.arange(1000)
        pm[10:100] = 10 + np.random.default_rng(124 + seed).permutation(90)
        return pm

    feat_cache = {}

    def b_features(geom, vals, perm):
        """geom_features of the values, z-scored over ALL values 10..99 (not per batch), so a batch does not rescale
        rare columns; the batch Gram is then only centred."""
        key = (geom, id(perm))
        if key not in feat_cache:
            F = geom_features(geom, np.arange(10, 100), perm).astype(np.float64)
            keep = F.std(0) > 1e-9
            feat_cache[key] = (F[:, keep] - F[:, keep].mean(0)) / F[:, keep].std(0)
        F = torch.tensor(feat_cache[key][np.asarray(vals) - 10], dtype=torch.float32, device=dev)
        F = F - F.mean(0)
        return F @ F.T

    def in_context_loss(prior, perm, rng):
        """Prior on the state of operand b (position 6) in a fresh draw of --prior_bs training prompts of the operation."""
        o, geom = prior.split("_", 1)
        ids, _, P = sets[(o, "train")]
        rows = rng.choice(len(ids), args.prior_bs, replace=False)
        forward(ids[torch.tensor(rows, device=dev)])
        K = b_features(geom, P[rows, 1], perm)
        return sum(1 - cka(store[L][:, 6], K) for L in layers_geo) / len(layers_geo)

    K_prefix = centred_gram(nl.features(np.arange(10, 100))[0], dev)

    def prefix_loss():
        forward(prefix)
        return sum(1 - cka(store[L][:, -1, :], K_prefix) for L in layers_geo) / len(layers_geo)

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        res = {}
        for (o, name), (ids, ans, _) in sets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs]).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[f"{o}/{name}"] = float((am == ans).float().mean())
        return res

    @torch.no_grad()
    def geometry() -> dict:
        """CKA of the operand states in each operation's TEST prompts to helix / log (true values)."""
        model.eval()
        rep = {}
        for o in TASKS:
            ids, _, P = sets[(o, "test")]
            ids, P = ids[:256], P[:256]
            forward(ids)
            for geom in ("helix", "log", "factor"):   # operand b only (a has not seen the operator)
                K = b_features(geom, P[:, 1], None)
                rep[f"{o}/b_{geom}"] = {f"L{L}": float(cka(store[L][:, 6], K)) for L in layers_geo}
        forward(prefix)
        rep["prefix/helix"] = {f"L{L}": float(cka(store[L][:, -1, :], K_prefix)) for L in layers_geo}
        return rep

    def reset(seed):
        torch.manual_seed(seed)
        from peft.tuners.lora import LoraLayer
        for m in model.modules():
            if isinstance(m, LoraLayer):
                m.reset_lora_parameters("default", True)

    base = {"eval": evaluate(), "geometry": geometry(), "n": {f"{o}/{n}": len(v[0]) for (o, n), v in sets.items()}}
    (out / "base.json").write_text(json.dumps(base, indent=1))
    print(f"base {base['eval']} ({time.time() - t0:.0f}s)", flush=True)
    for seed in seeds:
        perm = perm_for(seed)
        for arm in arms:
            path = out / f"run_{arm}_seed{seed}.json"
            if path.exists():
                continue
            active = [] if arm == "none" else arm.removesuffix("@norm").split("+")
            lam = args.lam / len(active) if arm.endswith("@norm") else args.lam
            params = [p for p in model.parameters() if p.requires_grad]
            grad_diag = None
            prior_rng = np.random.default_rng(3000 + seed)
            reset(seed)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
            rng = np.random.default_rng(1000 + seed)
            orders = {o: [] for o in TASKS}
            log = []
            model.train()
            for s in range(args.steps):
                opt.zero_grad(set_to_none=True)
                ce, geo = {}, {}
                for o in TASKS:
                    ids, ans, P = sets[(o, "train")]
                    if len(orders[o]) < args.bs:
                        orders[o] = list(rng.permutation(len(ids)))
                    rows = np.array([orders[o].pop() for _ in range(args.bs)])
                    bi = torch.tensor(rows, device=dev)
                    ce[o] = Fn.cross_entropy(forward(ids[bi]), ans[bi])
                for p in active:
                    geo[p] = prefix_loss() if p == "prefix_helix" else in_context_loss(p, perm, prior_rng)
                if s == 0:   # which terms pull the shared LoRA parameters in which direction
                    terms = {f"ce_{o}": v for o, v in ce.items()}
                    terms.update({f"prior_{p}": v for p, v in geo.items()})
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
                    log.append({"step": s, **{f"ce_{o}": float(v) for o, v in ce.items()}, **{f"geo_{p}": float(v) for p, v in geo.items()}})
            del opt
            ev = evaluate()
            path.write_text(json.dumps({"arm": arm, "active": active, "seed": seed, "lam": lam, "grad_diag": grad_diag, "rank": args.rank, "lr": args.lr,
                                        "steps": args.steps, "eval": ev, "geometry": geometry(), "log": log}, indent=1))
            print(f"{arm} seed{seed}: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items() if "test" in k) + f" ({time.time() - t0:.0f}s)", flush=True)
    summarize(out)


def summarize(out: Path) -> None:
    from scipy import stats
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    if not runs:
        return
    keys = [k for k in runs[0]["eval"] if "test" in k or k == "mul/train"]
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
