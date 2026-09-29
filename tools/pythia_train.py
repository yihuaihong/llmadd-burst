"""The manifold-shaping protocol of manifold_train.py on a second model family: Pythia (GPT-NeoX tokenizer).

Differences from the OLMo version, all forced by the tokenizer: the prompt is "Q: a + b =" and the answer is
the single token " <a+b>" (numbers carry their leading space; " 0".." 198" are single tokens, but only 544 of
100..999 are, so the three-digit set keeps only problems whose operands and answer are single tokens).
Same data (rng 0), same geometries, CKA loss at 1/8, 1/4, 3/8, 1/2 of the depth, LoRA on all linear layers.

    python tools/pythia_train.py --model <dir> --out <dir> --geoms none,helix,helix_shuf,digit --seeds 0,1,2
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
from manifold_train import cka, target_grams

GEOMS = ("none", "helix", "helix_shuf", "digit", "digit_shuf")


def layer_list(model):
    """The transformer blocks, whatever the family (model.model.layers, model.gpt_neox.layers, ...)."""
    best = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and name.split(".")[-1] == "layers":
            best = m if best is None or len(m) > len(best) else best
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geoms", default="none,helix,helix_shuf,digit")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default=None, help="default: 1/8, 1/4, 3/8, 1/2 of the depth")
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval_bs", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    geoms = args.geoms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    toks = tok.convert_ids_to_tokens(tok("Q: 23 + 45 =")["input_ids"])
    assert toks == ["Q", ":", "Ġ23", "Ġ+", "Ġ45", "Ġ="], toks
    SA, SB = 2, 4
    ans_id = np.full(1000, -1)
    for n in range(1000):
        ids = tok(" " + str(n), add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            ans_id[n] = ids[0]
    single = lambda n: ans_id[n] >= 0
    enc = lambda ps: nl.encode(tok, ps, dev)

    rng = np.random.default_rng(0)
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    pairs = pairs[rng.permutation(len(pairs))]
    train, test = pairs[:1500], pairs[1500:2500]
    trip = np.array([(a, b, c) for a in range(10, 100) for b in range(10, 100) for c in range(10, 100) if a + b + c <= 199])
    three = trip[rng.permutation(len(trip))][:400]
    big = np.array([(a, b) for a in range(100, 500) for b in range(100, 500)])
    big = big[rng.permutation(len(big))]
    big = np.array([p for p in big if single(p[0]) and single(p[1]) and single(p.sum())])[:400]
    evalsets = {
        "test": (enc([f"Q: {a} + {b} =" for a, b in test]), test.sum(1)),
        "train": (enc([f"Q: {a} + {b} =" for a, b in train]), train.sum(1)),
        "three_term": (enc([f"Q: {a} + {b} + {c} =" for a, b, c in three]), three.sum(1)),
        "three_digit": (enc([f"Q: {a} + {b} =" for a, b in big]), big.sum(1)),
    }
    evalsets = {k: (ids, torch.tensor(ans_id[ans], device=dev)) for k, (ids, ans) in evalsets.items()}
    train_ids, train_ans = evalsets["train"]
    prefix = enc([f"Q: {x}" for x in range(10, 100)])
    assert prefix.shape[1] == SA + 1
    perm = np.arange(100); perm[10:] = 10 + np.random.default_rng(123).permutation(90)
    grams = target_grams(perm, dev)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    nL = model.config.num_hidden_layers
    man_layers = [int(v) for v in args.layers.split(",")] if args.layers else sorted({max(1, round(nL * f)) for f in (1 / 8, 1 / 4, 3 / 8, 1 / 2)})
    targets = [n for n, _ in model.named_modules() if n.split(".")[-1] in ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
                                                                         "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")]
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                             target_modules=sorted({t.split(".")[-1] for t in targets})))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    blocks = layer_list(model)
    store = {}
    for L in man_layers:
        blocks[L].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))

    def forward(ids):
        return model(input_ids=ids).logits[:, -1].float()

    def geo_states():
        forward(prefix)
        return {L: store[L][:, SA, :] for L in man_layers}

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for name, (ids, ans) in evalsets.items():
            am = torch.cat([forward(ids[i:i + args.eval_bs]).argmax(-1) for i in range(0, len(ids), args.eval_bs)])
            res[name] = float((am == ans).float().mean())
        return res

    @torch.no_grad()
    def geometry_report():
        model.eval()
        return {f"L{L}": {"cka_" + G: float(cka(H, K)) for G, K in grams.items()} for L, H in geo_states().items()}

    base = {"eval": evaluate(), "geometry": geometry_report(), "layers": man_layers, "n_layers": nL, "n_three_digit": len(big)}
    (out / "base.json").write_text(json.dumps(base, indent=1))
    print(f"base {base['eval']} layers {man_layers} ({time.time() - t0:.0f}s)")
    for seed in seeds:
        for G in geoms:
            path = out / f"run_{G}_seed{seed}.json"
            if path.exists():
                continue
            torch.manual_seed(seed)
            from peft.tuners.lora import LoraLayer
            for m in model.modules():
                if isinstance(m, LoraLayer):
                    m.reset_lora_parameters("default", True)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
            order_rng = np.random.default_rng(1000 + seed)
            log = []
            model.train()
            for ep in range(args.epochs):
                order = order_rng.permutation(len(train_ids))
                ce_s = g_s = 0.0; nb = 0
                for i in range(0, len(order), args.bs):
                    bi = torch.tensor(order[i:i + args.bs], device=dev)
                    opt.zero_grad(set_to_none=True)
                    ce = Fn.cross_entropy(forward(train_ids[bi]), train_ans[bi])
                    Hs = geo_states() if G != "none" else None
                    if G == "none":
                        loss = ce; gl = torch.zeros(())
                    else:
                        gl = sum(1 - cka(H, grams[G]) for H in Hs.values()) / len(Hs)
                        loss = ce + args.lam * gl
                    loss.backward()
                    opt.step()
                    ce_s += float(ce.detach()); g_s += float(gl.detach()); nb += 1
                log.append({"epoch": ep, "ce": ce_s / nb, "geo_loss": g_s / nb})
            del opt
            ev = evaluate()
            path.write_text(json.dumps({"geom": G, "seed": seed, "lam": args.lam, "layers": man_layers, "train_log": log,
                                        "eval": ev, "geometry": geometry_report()}, indent=1))
            print(f"{G} seed{seed}: {ev} ({time.time() - t0:.0f}s)")
    summarize(out)


def summarize(out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    base = json.loads((out / "base.json").read_text())
    keys = list(base["eval"]); Lk = f"L{base['layers'][-1]}"
    lines = [f"# Pythia manifold shaping (layers {base['layers']} of {base['n_layers']}; three_digit n={base['n_three_digit']})\n\n",
             "| geom | seeds | " + " | ".join(keys) + f" | CKA@{Lk} helix / digit |\n", "|---|---|" + "---|" * (len(keys) + 1) + "\n",
             "| base | - | " + " | ".join(f"{base['eval'][k]:.3f}" for k in keys) + f" | {base['geometry'][Lk]['cka_helix']:.2f} / {base['geometry'][Lk]['cka_digit']:.2f} |\n"]
    for G in GEOMS:
        rs = [r for r in runs if r["geom"] == G]
        if not rs:
            continue
        cells = [f"{np.mean([r['eval'][k] for r in rs]):.3f} ± {np.std([r['eval'][k] for r in rs], ddof=1) if len(rs) > 1 else 0:.3f}" for k in keys]
        c = [np.mean([r["geometry"][Lk][f"cka_{g}"] for r in rs]) for g in ("helix", "digit")]
        lines.append(f"| {G} | {len(rs)} | " + " | ".join(cells) + f" | {c[0]:.2f} / {c[1]:.2f} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
