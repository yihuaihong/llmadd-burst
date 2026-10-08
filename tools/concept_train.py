"""Manifold-shaping fine-tuning beyond two-digit numbers: other structured concepts (same protocol as
manifold_train.py, LoRA + CKA geometry loss, paired seeds).

Domains (every item is ONE token after a space, and it is the 3rd token of every prompt, so its state in the
geometry prefix "Q: <item>" equals its state inside every task prompt):
  days     cyclic, 7    "Q: Friday plus 3 days is" -> " Monday"      targets: circle | circle_shuf | line (wrong topology)
  months   cyclic, 12   "Q: March minus 5 months is" -> " October"
  letters  linear, 26   "Q: F plus 3 letters is" -> " I"            targets: line | line_shuf | circle (wrong topology)
  numwords number       "Q: seven + twelve = " -> "19"  (one..twenty, thirty..ninety)   targets: helix | helix_shuf | digit
  roman    number       "Q: VII + XII = " -> "19"       (I..XVI)
The number-word and Roman domains keep the geometry of the digit domain but change the surface form (rarer
tokens): does their manifold form later, and does the benefit window close later?

Prompt-state targets (cyclic domains; P2/P3, after "Arithmetic in the Wild", arXiv 2605.01148, where Llama adds
days and months on its base-10 number line instead of on a circle). Shaped: the FINAL-token state of a fixed
batch of 96 training prompts, not the item state:
  sum_helix       layers 3/8, 1/2, 5/8 of the depth -> helix of the signed pre-mod sum s = i +- k (number route)
  sum_helix_shuf  the same with s permuted (same basis, wrong values)
  out_circle      layers 3/4, 7/8 -> circle of the answer index (i +- k) mod n (result on the cycle)
--route none,circle runs the number-route read-outs of number_route.py (sum route, operand route, item-number
alignment) on the base model and on every trained model of the listed arms, i.e. on models that solve the task.

    python tools/concept_train.py --model <dir> --out <dir> --domain days --seeds 0,1,2
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
from manifold_train import cka, decoder_layers

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
LETTERS = [chr(c) for c in range(65, 91)]
NUMWORDS = dict(zip(range(1, 21), "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                    "fifteen sixteen seventeen eighteen nineteen twenty".split()))
NUMWORDS.update(dict(zip(range(30, 100, 10), "thirty forty fifty sixty seventy eighty ninety".split())))
ROMAN = dict(zip(range(1, 17), "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI".split()))

DOMAINS = {
    "days": dict(kind="cyclic", items=DAYS, unit="days", kmax=40, targets=("circle", "circle_shuf", "line")),
    "months": dict(kind="cyclic", items=MONTHS, unit="months", kmax=40, targets=("circle", "circle_shuf", "line")),
    "letters": dict(kind="linear", items=LETTERS, unit="letters", kmax=12, targets=("line", "line_shuf", "circle")),
    "numwords": dict(kind="number", values=NUMWORDS, targets=("helix", "helix_shuf", "digit")),
    "roman": dict(kind="number", values=ROMAN, targets=("helix", "helix_shuf", "digit")),
}


def items_and_values(dom: dict):
    if dom["kind"] == "number":
        vals = np.array(sorted(dom["values"]))
        return [dom["values"][v] for v in vals], vals
    return list(dom["items"]), np.arange(len(dom["items"]))


def target_features(dom: dict, G: str, vals: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Ideal coordinates of the domain's items (rows in item order); '_shuf' = same basis, permuted items."""
    idx = np.arange(len(vals))
    if G.endswith("_shuf"):
        idx = perm; G = G[: -len("_shuf")]
    v, n = vals[idx], len(vals)
    if G == "circle":
        return np.stack([np.cos(2 * np.pi * v / n), np.sin(2 * np.pi * v / n)], 1)
    if G == "line":
        return v[:, None].astype(float)
    if G == "helix":
        return nl.features(v)[0]
    if G == "digit":
        return np.concatenate([np.eye(10)[v // 10], np.eye(10)[v % 10]], 1)
    raise ValueError(G)


def centred_gram(F: np.ndarray, dev) -> torch.Tensor:
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    Ft = torch.tensor(F, dtype=torch.float32, device=dev)
    return Ft @ Ft.T


PROMPT_GEOMS = ("sum_helix", "sum_helix_shuf", "out_circle")


def parse_prompt(p: str, items: list):
    """'Q: Friday plus 3 days is' -> (item index, signed pre-mod sum i +- k)."""
    w = p.split()
    i, k = items.index(w[1]), int(w[3])
    return i, (i + k if w[2] == "plus" else i - k)


def prompt_targets(dom: dict, items: list, data: dict, evalsets: dict, rng_seed: int, dev):
    """Target Grams of the prompt-state geometries on a fixed batch of 96 prompts each from train and test, and a
    builder for the targets of any batch of train prompts."""
    n, K = len(items), dom["kmax"]
    s_all = np.arange(-K, n + K)
    s_perm = dict(zip(s_all, np.random.default_rng(321).permutation(s_all)))
    def grams(s):
        j = s % n
        return {
            "sum_helix": centred_gram(nl.features(s)[0], dev),
            "sum_helix_shuf": centred_gram(nl.features(np.array([s_perm[v] for v in s]))[0], dev),
            "out_circle": centred_gram(np.stack([np.cos(2 * np.pi * j / n), np.sin(2 * np.pi * j / n)], 1), dev),
        }

    rng = np.random.default_rng(rng_seed)
    targets, batch = {}, {}
    for which in ("train", "test"):
        rows = np.sort(rng.choice(len(data[which]), 96, replace=False))
        targets[which] = grams(np.array([parse_prompt(data[which][r][0], items)[1] for r in rows]))
        batch[which] = evalsets[which][0][torch.tensor(rows, device=evalsets[which][0].device)]
    s_train = np.array([parse_prompt(p, items)[1] for p, _ in data["train"]])
    return targets, batch, (lambda rows: grams(s_train[rows]))   # last: targets of any train rows (--presample)


def build_data(dom: dict, items: list, vals: np.ndarray, rng) -> dict:
    """{split: [(prompt, answer_string)]}: train / test (unseen combinations) and, for cyclic domains, ood (larger shifts)."""
    probs = []
    if dom["kind"] == "number":
        for a in vals:
            for b in vals:
                probs.append((f"Q: {dom['values'][a]} + {dom['values'][b]} = ", str(a + b)))
        probs = [probs[i] for i in rng.permutation(len(probs))]
        ntr = int(0.6 * len(probs))
        return {"train": probs[:ntr], "test": probs[ntr:]}
    n, K = len(items), dom["kmax"]

    def make(ks):
        out = []
        for i in range(n):
            for k in ks:
                for op, s in (("plus", 1), ("minus", -1)):
                    j = i + s * k
                    if dom["kind"] == "cyclic":
                        j %= n
                    elif not 0 <= j < n:
                        continue
                    out.append((f"Q: {items[i]} {op} {k} {dom['unit']} is", " " + items[j]))
        return [out[t] for t in rng.permutation(len(out))]
    probs = make(range(1, K + 1))
    ntr = int(0.6 * len(probs))
    data = {"train": probs[:ntr], "test": probs[ntr:]}
    if dom["kind"] == "cyclic":
        data["ood"] = make(range(K + 1, 2 * K + 1))[:300]
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", choices=tuple(DOMAINS), required=True)
    ap.add_argument("--geoms", default=None, help="default: none + the domain's three targets")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--plam", type=float, default=None, help="weight of the prompt-state geometries (default: --lam)")
    ap.add_argument("--presample", action="store_true", help="prompt-state geometries on a fresh batch of 96 train prompts every step (default: one fixed batch)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=280, help="optimizer steps per run (epochs = ceil(steps * bs / n_train))")
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--sum_layers", default=None, help="final-token layers of sum_helix* (default 3/8, 1/2, 5/8 of the depth)")
    ap.add_argument("--out_layers", default=None, help="final-token layers of out_circle (default 3/4, 7/8 of the depth)")
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--route", default="", help="arms (comma list, or all) analysed with number_route.analyze after training (days / months / letters)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dom = DOMAINS[args.domain]
    geoms = (args.geoms.split(",") if args.geoms else ["none", *dom["targets"]])
    assert all(G == "none" or G in dom["targets"] or (G in PROMPT_GEOMS and dom["kind"] == "cyclic") for G in geoms), geoms
    seeds = [int(s) for s in args.seeds.split(",")]
    man_layers = [int(v) for v in args.layers.split(",")]
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    items, vals = items_and_values(dom)
    enc = lambda ps: nl.encode(tok, ps, dev)
    prefix = enc([f"Q: {w}" for w in items])
    assert prefix.shape[1] == 3, f"every item must be one token: {prefix.shape}"
    SLOT = 2
    rng = np.random.default_rng(0)
    data = build_data(dom, items, vals, rng)
    evalsets = {}
    for name, probs in data.items():
        ids = enc([p for p, _ in probs])
        ans =[tok(a, add_special_tokens=False)["input_ids"] for _, a in probs]
        assert all(len(a) == 1 for a in ans), "answers must be single tokens"
        evalsets[name] = (ids, torch.tensor([a[0] for a in ans], device=dev))
    item_of = {w: i for i, w in enumerate(items)}
    for name, probs in data.items():   # the item is always the 3rd token (index SLOT) of the prompt
        first = [tok.convert_ids_to_tokens(int(evalsets[name][0][r, SLOT])) for r in range(len(probs))]
        assert all(f.lstrip("Ġ") in item_of for f in first), (name, first[:5])
    train_ids, train_ans = evalsets["train"]
    perm = np.random.default_rng(123).permutation(len(items))
    targets = {G: centred_gram(target_features(dom, G, vals, perm), dev) for G in dom["targets"]}

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    layers = decoder_layers(model)
    nL = len(layers)
    sum_layers = [int(v) for v in args.sum_layers.split(",")] if args.sum_layers else [round(nL * f) for f in (3 / 8, 1 / 2, 5 / 8)]
    out_layers = [int(v) for v in args.out_layers.split(",")] if args.out_layers else [round(nL * f) for f in (3 / 4, 7 / 8)]
    prompt_on = dom["kind"] == "cyclic"
    if prompt_on:
        ptargets, pbatch, pgrams = prompt_targets(dom, items, data, evalsets, rng_seed=55, dev=dev)
    store: dict = {}
    for L in sorted(set(man_layers) | (set(sum_layers) | set(out_layers) if prompt_on else set())):
        layers[L].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))
    emb_mod = model.get_input_embeddings()
    item_tok = prefix[:, SLOT]

    def forward(ids):
        return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()

    def geo_states():
        forward(prefix)
        return {L: store[L][:, SLOT, :] for L in man_layers}

    def prompt_states(which):
        forward(pbatch[which])
        return {L: store[L][:, -1, :] for L in sorted(set(sum_layers) | set(out_layers))}

    presample_rng = np.random.default_rng(0)   # re-seeded per run below

    def geo_loss(G):
        if G in PROMPT_GEOMS:
            Ls = out_layers if G == "out_circle" else sum_layers
            if args.presample:
                rows = np.sort(presample_rng.choice(len(train_ids), 96, replace=False))
                forward(train_ids[torch.tensor(rows, device=dev)])
                K = pgrams(rows)[G]
                return sum(1 - cka(store[L][:, -1, :], K) for L in Ls) / len(Ls)
            Hs = prompt_states("train")
            return sum(1 - cka(Hs[L], ptargets["train"][G]) for L in Ls) / len(Ls)
        Hs = geo_states()
        return sum(1 - cka(H, targets[G]) for H in Hs.values()) / len(Hs)

    @torch.no_grad()
    def geometry_report() -> dict:
        model.eval()
        Hs = geo_states(); Hs["emb"] = emb_mod.weight[item_tok].float()
        rep = {(f"L{L}" if L != "emb" else "emb"): {"cka_" + G: float(cka(H, K)) for G, K in targets.items()} for L, H in Hs.items()}
        if prompt_on:   # final-token geometry on the fixed train batch and on 96 test prompts
            rep["prompt"] = {}
            for which in ("train", "test"):
                Ps = prompt_states(which)
                rep["prompt"][which] = {f"L{L}": {"cka_" + G: float(cka(H, K)) for G, K in ptargets[which].items()} for L, H in Ps.items()}
            rep["prompt"]["sum_layers"], rep["prompt"]["out_layers"] = sum_layers, out_layers
        return rep

    @torch.no_grad()
    def evaluate(preds=None) -> dict:
        model.eval()
        res = {}
        for name, (ids, ans) in evalsets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs]).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[name] = float((am == ans).float().mean())
            if preds is not None and name != "train":
                preds[name] = tok.convert_ids_to_tokens(am.tolist())
        return res

    def reset(seed):
        torch.manual_seed(seed)
        from peft.tuners.lora import LoraLayer
        for m in model.modules():
            if isinstance(m, LoraLayer):
                m.reset_lora_parameters("default", True)

    route_arms = set(geoms if args.route == "all" else filter(None, args.route.split(",")))
    if route_arms:
        from number_route import DOMAINS as ROUTE_DOMAINS, analyze
        assert args.domain in ROUTE_DOMAINS and args.domain != "numwords", "--route: days, months or letters"
        route_layers = list(range(2, nL, 2))

    def route(tag):
        model.eval()
        return analyze(model, tok, [args.domain], route_layers, dev, np.random.default_rng(0), shots=0,
                       log=lambda m: print(f"{tag} {m}", flush=True))

    base = {"domain": args.domain, "items": items, "n": {k: len(v) for k, v in data.items()},
            "eval": evaluate(), "geometry": geometry_report(), "examples": {k: v[:3] for k, v in data.items()}}
    if route_arms:
        base["route"] = route("base")
    (out / "base.json").write_text(json.dumps(base, indent=1))
    print(f"base: {base['eval']} geometry L16/emb: {base['geometry'].get('L16')} {base['geometry']['emb']} ({time.time() - t0:.0f}s)")
    epochs = max(1, math.ceil(args.steps * args.bs / len(train_ids)))
    for seed in seeds:
        for G in geoms:
            path = out / f"run_{G}_seed{seed}.json"
            if path.exists():
                continue
            reset(seed)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
            order_rng = np.random.default_rng(1000 + seed)
            presample_rng = np.random.default_rng(2000 + seed)
            log = []
            model.train()
            for ep in range(epochs):
                order = order_rng.permutation(len(train_ids))
                ce_s = g_s = 0.0; nb = 0
                for i in range(0, len(order), args.bs):
                    bi = torch.tensor(order[i:i + args.bs], device=dev)
                    opt.zero_grad(set_to_none=True)
                    ce = Fn.cross_entropy(forward(train_ids[bi]), train_ans[bi])
                    if G == "none":
                        loss = ce
                        with torch.no_grad():
                            gl = geo_loss(dom["targets"][0])
                    else:
                        gl = geo_loss(G)
                        loss = ce + (args.plam if (G in PROMPT_GEOMS and args.plam is not None) else args.lam) * gl
                    loss.backward()
                    opt.step()
                    ce_s += float(ce.detach()); g_s += float(gl.detach()); nb += 1
                log.append({"epoch": ep, "ce": ce_s / nb, "geo_loss": g_s / nb})
            del opt
            preds = {} if args.save_preds else None
            ev = evaluate(preds)
            rt = route(f"{G} seed{seed}") if G in route_arms else None
            path.write_text(json.dumps({"domain": args.domain, "geom": G, "seed": seed, "lam": args.lam, "plam": args.plam, "presample": args.presample, "lr": args.lr,
                                        "epochs": epochs, "layers": man_layers, "train_log": log, "eval": ev,
                                        "geometry": geometry_report(), "preds": preds, "route": rt}, indent=1))
            print(f"{args.domain} {G} seed{seed}: {ev} ({time.time() - t0:.0f}s)")
    summarize(out)


def summarize(out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    base = json.loads((out / "base.json").read_text())
    keys = list(base["eval"])
    tg = [k[4:] for k in base["geometry"]["emb"]]
    ideal = tg[0]
    Lk = [k for k in base["geometry"] if k.startswith("L")][-1]
    lines = [f"# concept shaping: {base['domain']} (n = {base['n']})\n\n",
             f"base geometry: emb CKA({ideal}) {base['geometry']['emb']['cka_' + ideal]:.2f}, {Lk} CKA({ideal}) {base['geometry'][Lk]['cka_' + ideal]:.2f}\n\n",
             "| geom | seeds | " + " | ".join(keys) + f" | CKA@{Lk} " + " / ".join(tg) + " |\n", "|---|---|" + "---|" * (len(keys) + 1) + "\n",
             "| base | - | " + " | ".join(f"{base['eval'][k]:.3f}" for k in keys) + " | " + " / ".join(f"{base['geometry'][Lk]['cka_' + g]:.2f}" for g in tg) + " |\n"]
    for G in dict.fromkeys(r["geom"] for r in runs):
        rs = [r for r in runs if r["geom"] == G]
        cells = []
        for k in keys:
            v = np.array([r["eval"][k] for r in rs])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        c = " / ".join(f"{np.mean([r['geometry'][Lk]['cka_' + g] for r in rs]):.2f}" for g in tg)
        if "prompt" in rs[0]["geometry"]:   # final-token geometry on held-out (test) prompts
            pr = [r["geometry"]["prompt"] for r in rs]
            sL, oL = f"L{pr[0]['sum_layers'][1]}", f"L{pr[0]['out_layers'][0]}"
            c += (f"; test prompts: sum_helix@{sL} {np.mean([q['test'][sL]['cka_sum_helix'] for q in pr]):.2f}"
                  f" out_circle@{oL} {np.mean([q['test'][oL]['cka_out_circle'] for q in pr]):.2f}")
        lines.append(f"| {G} | {len(rs)} | " + " | ".join(cells) + f" | {c} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
