"""Manifold-regularised fine-tuning of an OLMo-2 checkpoint on two-digit addition.

Question: if fine-tuning also pulls the operand representations onto an ideal number manifold, does the
model generalise better (held-out pairs, three-term, three-digit) than with the task loss alone?

Loss:  CE(answer token) + lam * L_geo
L_geo: the GEOMETRY of the operand representation, not a readout of it. Every step the 90 prefixes "Q: x"
       (x = 10..99) are run through the model; at each layer in `--layers` the states of x form H [90, d],
       and L_geo = mean_L (1 - CKA(H, F_G)), with F_G [90, k] the ideal coordinates of 10..99 and linear
       CKA on centred Gram matrices. CKA is invariant only to rotations and isotropic scaling, so it is
       satisfied only if the representation itself has the shape of the ideal manifold.
       (v1 used ||(h - mu) R - F||^2 with a readout R fitted in 4096-d; 90 points can be mapped linearly
       onto ANY target there - shuffled included - so v1 constrained nothing. Found on s1-10k, 2026-09-27.)
Geometries G: none (task only) | helix | helix_shuf | digit | digit_shuf. "_shuf" uses the same basis on a
       fixed permutation of 10..99: same dimensionality and strength, wrong numbers.
Methods (--mode): lora (r=8, all linear layers) | reft (low-rank edit of the operand states at the
       manifold layers, base frozen) | full (all weights, fp32 master + AdamW; needs ~120 GB, e.g. one H200).

Every (geometry, seed) run starts from the same base weights; for a given seed all geometries share the
data order and the initialisation of the trainable parameters, so comparisons are paired. One JSON per
run under --out; existing runs are skipped (restartable).

    python tools/manifold_train.py --model <dir> --out <dir> --mode lora --geoms none,helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
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

GEOMS = ("none", "helix", "helix_shuf", "digit", "digit_shuf")


def two_prompt(a, b): return f"Q: {a} + {b} = "
def three_prompt(a, b, c): return f"Q: {a} + {b} + {c} = "
def terse_prompt(a, b): return f"{nl.PREFIX}{a}+{b}="


def geom_features(G: str, x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if G.endswith("_shuf"):
        x = perm[x]; G = G[: -len("_shuf")]
    if G == "helix":
        return nl.features(x)[0]
    if G == "digit":
        return np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1)
    raise ValueError(G)


def cka(H: torch.Tensor, K_F: torch.Tensor) -> torch.Tensor:
    """Linear CKA between states H [n, d] and a centred target Gram K_F [n, n] (differentiable in H)."""
    H = H.float() - H.float().mean(0, keepdim=True)
    K_H = H @ H.T
    return (K_H * K_F).sum() / (K_H.norm() * K_F.norm() + 1e-12)


def target_grams(perm: np.ndarray, dev) -> dict:
    """Centred Gram matrices of the ideal coordinates of 10..99 for every geometry."""
    out = {}
    for G in ("helix", "helix_shuf", "digit", "digit_shuf"):
        F = geom_features(G, np.arange(10, 100), perm)
        F = F[:, F.std(0) > 1e-9]
        F = (F - F.mean(0)) / F.std(0)
        Ft = torch.tensor(F, dtype=torch.float32, device=dev)
        out[G] = Ft @ Ft.T
    return out


def decoder_layers(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "Olmo2DecoderLayer"]


class Reft(torch.nn.Module):
    """Low-rank additive edit of selected token states at one layer: h + (h Wd^T + b) Wu^T (Wu starts at 0)."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.down = torch.nn.Linear(d, r)
        self.up = torch.nn.Linear(r, d, bias=False)

    def reset(self) -> None:
        torch.nn.init.normal_(self.down.weight, std=self.down.in_features ** -0.5)
        torch.nn.init.zeros_(self.down.bias)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.down(h.float())).to(h.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("lora", "reft", "full"), required=True)
    ap.add_argument("--geoms", default=",".join(GEOMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--train_pairs", type=int, default=1500)
    ap.add_argument("--test_pairs", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lr = args.lr or {"lora": 1e-4, "reft": 1e-3, "full": 1e-5}[args.mode]
    geoms = args.geoms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    man_layers = [int(v) for v in args.layers.split(",")]
    t0 = time.time()

    # ---------------------------------------------------------------- model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    base_dtype = torch.float32 if (args.mode == "full" or dev == "cpu") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype).to(dev)
    model.config.use_cache = False
    layers = decoder_layers(model)
    d = model.config.hidden_size
    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    toks = tok.convert_ids_to_tokens(tok(two_prompt(23, 45))["input_ids"])
    assert toks[-6:] == ["23", "Ġ+", "Ġ", "45", "Ġ=", "Ġ"], toks
    slot_pos = {"A": len(toks) - 6, "B": len(toks) - 3}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (args.mode == "full" and dev == "cuda") else torch.autocast("cpu", enabled=False)

    reft_mods = None
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        layers = decoder_layers(model)
    elif args.mode == "reft":
        for p in model.parameters():
            p.requires_grad_(False)
        reft_mods = torch.nn.ModuleDict({str(L): Reft(d, args.rank) for L in man_layers}).to(dev)
    else:
        init_state = {k: v.detach().to("cpu", torch.bfloat16).clone() for k, v in model.state_dict().items()}

    def trainable():
        if args.mode == "reft":
            return list(reft_mods.parameters())
        return [p for p in model.parameters() if p.requires_grad]

    def reset(seed: int) -> None:
        torch.manual_seed(seed)
        if args.mode == "lora":
            from peft.tuners.lora import LoraLayer
            for m in model.modules():
                if isinstance(m, LoraLayer):
                    m.reset_lora_parameters("default", True)
        elif args.mode == "reft":
            for m in reft_mods.values():
                m.reset()
        else:
            model.load_state_dict({k: v.to(torch.float32) for k, v in init_state.items()})

    # ---------------------------------------------------------------- hooks: capture (+ ReFT edit) at manifold layers
    store: dict = {}
    reft_on = {"v": False}

    def make_hook(L: int):
        def hook(_m, _i, o):
            h = nl._hidden(o)
            if reft_mods is not None and reft_on["v"]:
                h0, h = h, h.clone()          # read from h0, write into the copy (autograd-safe)
                for p in slot_pos.values():
                    if p < h.shape[1]:
                        h[:, p, :] = reft_mods[str(L)](h0[:, p, :])
                o = nl._with_hidden(o, h)
            store[L] = h
            return o
        return hook

    for L in man_layers:
        layers[L].register_forward_hook(make_hook(L))

    def forward(ids: torch.Tensor, reft: bool) -> torch.Tensor:
        reft_on["v"] = reft
        with autocast:
            return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()

    # ---------------------------------------------------------------- data
    rng = np.random.default_rng(0)
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    pairs = pairs[rng.permutation(len(pairs))]
    train, test = pairs[: args.train_pairs], pairs[args.train_pairs: args.train_pairs + args.test_pairs]
    for j in range(2):   # per-number means are taken per slot, so every value must occur in every slot
        assert set(np.unique(train[:, j])) >= set(range(10, 100)), f"slot {j}: some operand value never occurs in training"
    trip = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 199])
    three = trip[rng.permutation(len(trip))][:400]
    big = np.array([(a, b) for a in range(100, 500) for b in range(100, 500)])
    big = big[rng.permutation(len(big))][:400]
    perm = np.arange(100); perm[10:] = 10 + np.random.default_rng(123).permutation(90)
    enc = lambda ps: nl.encode(tok, ps, dev)
    evalsets = {
        "test": (enc([two_prompt(*x) for x in test]), test.sum(1)),
        "train": (enc([two_prompt(*x) for x in train]), train.sum(1)),
        "three_term": (enc([three_prompt(*x) for x in three]), three.sum(1)),
        "three_digit": (enc([two_prompt(*x) for x in big]), big.sum(1)),
        "test_terse": (enc([terse_prompt(*x) for x in test]), test.sum(1)),
    }
    train_ids = evalsets["train"][0]

    @torch.no_grad()
    def evaluate(reft: bool) -> dict:
        model.eval()
        res = {}
        for name, (ids, ans) in evalsets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs], reft).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[name] = float((am == num_ids[torch.tensor(ans, device=dev)]).float().mean())
        return res

    @torch.no_grad()
    def number_means(reft: bool) -> dict:
        """{(L, slot): [90, d]} per-number mean state over the training prompts."""
        model.eval()
        sums = {(L, s): torch.zeros(90, d, device=dev) for L in man_layers for s in slot_pos}
        cnt = {s: torch.zeros(90, device=dev) for s in slot_pos}
        for i in range(0, len(train_ids), args.eval_bs):
            forward(train_ids[i:i + args.eval_bs], reft)
            chunk = train[i:i + args.eval_bs]
            for j, s in enumerate(slot_pos):
                idx = torch.tensor(chunk[:, j] - 10, device=dev)
                for L in man_layers:
                    sums[(L, s)].index_add_(0, idx, store[L][:, slot_pos[s], :].float())
                cnt[s].index_add_(0, idx, torch.ones(len(idx), device=dev))
        return {k: (v / cnt[k[1]][:, None]).cpu().numpy() for k, v in sums.items()}

    grams = target_grams(perm, dev)
    num_prefix = enc([f"Q: {x}" for x in range(10, 100)])            # A-slot state of x depends only on this prefix
    assert num_prefix.shape[1] == slot_pos["A"] + 1

    def geo_cka(reft: bool) -> dict:
        """{L: H [90, d]} for the 90 numbers (with grad)."""
        forward(num_prefix, reft)
        return {L: store[L][:, slot_pos["A"], :] for L in man_layers}

    @torch.no_grad()
    def geometry_report(reft: bool) -> dict:
        model.eval()
        Hs = geo_cka(reft)
        return {f"L{L}": {"cka_" + G: float(cka(Hs[L], grams[G])) for G in grams} for L in man_layers}

    base_eval = evaluate(reft=False)
    base_means = number_means(reft=False)
    (out / "base.json").write_text(json.dumps({"eval": base_eval, "manifold": diagnostics(base_means, None, slot_pos),
                                               "geometry": geometry_report(reft=False)}, indent=1))
    print(f"base: {base_eval} ({time.time() - t0:.0f}s)")

    for seed in seeds:
        for G in geoms:
            path = out / f"run_{G}_seed{seed}.json"
            if path.exists():
                continue
            reset(seed)
            params = trainable()
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, **({"fused": True} if (args.mode == "full" and dev == "cuda") else {}))
            order_rng = np.random.default_rng(1000 + seed)
            log = []
            model.train()
            for ep in range(args.epochs):
                order = order_rng.permutation(len(train))
                ce_s = man_s = 0.0; nb = 0
                for i in range(0, len(order), args.bs):
                    bi = order[i:i + args.bs]
                    chunk = train[bi]
                    logits = forward(train_ids[torch.tensor(bi, device=dev)], reft=True)
                    ce = Fn.cross_entropy(logits, num_ids[torch.tensor(chunk.sum(1), device=dev)])
                    if G == "none":
                        with torch.no_grad():
                            Hs = geo_cka(reft=True)
                            ml = sum(1 - cka(Hs[L], grams["helix"]) for L in man_layers) / len(man_layers)
                        loss = ce
                    else:
                        Hs = geo_cka(reft=True)
                        ml = sum(1 - cka(Hs[L], grams[G]) for L in man_layers) / len(man_layers)
                        loss = ce + args.lam * ml
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    ce_s += float(ce); man_s += float(ml); nb += 1
                log.append({"epoch": ep, "ce": ce_s / nb, "geo_loss": man_s / nb})
            del opt
            ev = evaluate(reft=True)
            diag = diagnostics(number_means(reft=True), None, slot_pos)
            path.write_text(json.dumps({"mode": args.mode, "loss": "v2_cka", "geom": G, "seed": seed, "lam": args.lam, "lr": lr,
                                        "layers": man_layers, "train_log": log, "eval": ev, "manifold": diag,
                                        "geometry": geometry_report(reft=True)}, indent=1))
            print(f"{args.mode} {G} seed{seed}: {ev} ({time.time() - t0:.0f}s)")
    summarize(out)


def diagnostics(means: dict, targets: dict, slot_pos: dict) -> dict:
    """Held-out-number R^2 of the helix and digit geometries on the per-number mean states (slot A)."""
    out = {}
    x = np.arange(10, 100)
    for (L, s), X in means.items():
        if s != "A":
            continue
        out[f"L{L}"] = {"r2_helix": nl.cv_r2(nl.features(x)[0], X),
                        "r2_digit": nl.cv_r2(np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1), X)}
    return out


def summarize(out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    if not runs:
        return
    base = json.loads((out / "base.json").read_text())
    keys = list(runs[0]["eval"])
    lines = [f"# manifold fine-tuning ({runs[0]['mode']}, lam {runs[0]['lam']}, lr {runs[0]['lr']}, layers {runs[0]['layers']})\n\n",
             "| geom | n seeds | " + " | ".join(keys) + " | r2 helix / digit (first layer) |\n", "|---|---|" + "---|" * (len(keys) + 1) + "\n",
             "| base | - | " + " | ".join(f"{base['eval'][k]:.3f}" for k in keys) + " | - |\n"]
    for G in GEOMS:
        rs = [r for r in runs if r["geom"] == G]
        if not rs:
            continue
        cells = []
        for k in keys:
            v = np.array([r["eval"][k] for r in rs])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        first = next(iter(rs[0]["manifold"]))
        r2 = np.mean([[r["manifold"][first]["r2_helix"], r["manifold"][first]["r2_digit"]] for r in rs], 0)
        ck = ""
        if "geometry" in rs[0]:
            gl = list(rs[0]["geometry"])[-1]
            c = np.mean([[r["geometry"][gl]["cka_helix"], r["geometry"][gl]["cka_digit"]] for r in rs], 0)
            ck = f" CKA@{gl} helix {c[0]:.2f} digit {c[1]:.2f}"
        lines.append(f"| {G} | {len(rs)} | " + " | ".join(cells) + f" | {r2[0]:.2f} / {r2[1]:.2f}{ck} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
