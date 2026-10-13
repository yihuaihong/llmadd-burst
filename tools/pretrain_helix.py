"""From-scratch pretraining with a helix prior: does shaping the number manifold during PRETRAINING make two-digit
addition emerge earlier / better, and at what cost to language modelling?

Data: FineWeb-Edu tokens (pretrain_data.py) with one arithmetic line appended to every training sequence (~2% of
tokens): "Q: a + b = c" and three other surface formats, a, b in 10..99, over the TRAIN pairs only (1000 random
pairs are held out and never seen). Model: a small OLMo-2 architecture (default d 384, 6 layers) from scratch.
Arms (--arm):
  none         language-modelling loss only
  helix        + lam * (1 - CKA) between the last-token states of "Q: x" (x = 10..99) at --geo_layers and the helix
  helix_shuf   the same with the numbers permuted (wrong geometry, same strength)
  helix_window helix only for the steps in [--window start, end) (fractions of training)
Logged every --eval_every steps: held-out addition accuracy ("Q: a + b = " -> answer token; and "a plus b equals "),
train-pair accuracy, validation LM loss on held-out FineWeb text, CKA(helix) of the number states.

    python tools/pretrain_helix.py --data <dir> --tokenizer <olmo2 dir> --out <dir> --arm helix --seed 0
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

FORMATS = ("Q: {a} + {b} = ", "{a} plus {b} equals ", "The sum of {a} and {b} is ", "Compute {a} + {b}. Result: ")


def cka(H: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Linear CKA between states H [n, d] and a centred target Gram K [n, n]."""
    H = H.float() - H.float().mean(0)
    G = H @ H.T
    return (G * K).sum() / (G.norm() * K.norm() + 1e-8)


def centred_gram(F: np.ndarray, dev) -> torch.Tensor:
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    t = torch.tensor(F, dtype=torch.float32, device=dev)
    t = t - t.mean(0)
    return t @ t.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=("none", "helix", "helix_shuf", "helix_window"), required=True)
    ap.add_argument("--window", default="0.1,0.4", help="helix_window: start,end as fractions of the steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_seed", type=int, default=None, help="model initialisation seed (default --seed)")
    ap.add_argument("--data_seed", type=int, default=None, help="data order seed (default --seed)")
    ap.add_argument("--tag", default="", help="suffix of the run file name")
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=float, default=3e8, help="training tokens")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--geo_layers", default=None, help="default: n_layers/3 and 2 n_layers/3")
    ap.add_argument("--eval_every", type=int, default=250)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    init_seed = args.seed if args.init_seed is None else args.init_seed
    data_seed = args.seed if args.data_seed is None else args.data_seed
    path = out / f"run_{args.arm}_seed{args.seed}{args.tag}.json"
    torch.manual_seed(init_seed); np.random.seed(init_seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint32, mode="r")
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint32, mode="r")

    # ------------------------------------------------------------ arithmetic (held-out pairs never appear)
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    pairs = pairs[np.random.default_rng(0).permutation(len(pairs))]
    test, trainp = pairs[:1000], pairs[1000:]
    lines = [[tok(f.format(a=a, b=b) + f"{a + b}\n", add_special_tokens=False)["input_ids"] for f in FORMATS] for a, b in trainp]
    enc = lambda ps: nl.encode(tok, ps, dev)
    ans_tok = lambda P: torch.tensor([tok(str(a + b), add_special_tokens=False)["input_ids"][0] for a, b in P], device=dev)
    evalsets = {"test": (enc([FORMATS[0].format(a=a, b=b) for a, b in test]), ans_tok(test)),
                "test_plus": (enc([FORMATS[1].format(a=a, b=b) for a, b in test]), ans_tok(test)),
                "train": (enc([FORMATS[0].format(a=a, b=b) for a, b in trainp[:500]]), ans_tok(trainp[:500]))}
    num_prefix = enc([f"Q: {x}" for x in range(10, 100)])
    xs = np.arange(10, 100)
    K = {"helix": centred_gram(nl.features(xs)[0], dev),
         "helix_shuf": centred_gram(nl.features(xs[np.random.default_rng(124).permutation(90)])[0], dev)}

    # ------------------------------------------------------------ model
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=args.d_model, intermediate_size=args.d_ff,
                      num_hidden_layers=args.n_layers, num_attention_heads=args.n_heads, num_key_value_heads=args.n_heads,
                      max_position_embeddings=args.seq, tie_word_embeddings=True,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = Olmo2ForCausalLM(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    geo_layers = [int(v) for v in args.geo_layers.split(",")] if args.geo_layers else [args.n_layers // 3, 2 * args.n_layers // 3]
    store = {}
    for L in geo_layers:
        model.model.layers[L - 1].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))
    steps = int(args.tokens // (args.bs * args.seq))
    warm = max(1, steps // 100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=(dev == "cuda"))
    sched = lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    w0, w1 = (float(v) for v in args.window.split(","))
    geo_on = {"none": lambda s: False, "helix": lambda s: True, "helix_shuf": lambda s: True,
              "helix_window": lambda s: w0 * steps <= s < w1 * steps}[args.arm]
    target = K["helix_shuf" if args.arm == "helix_shuf" else "helix"]
    rng = np.random.default_rng(1000 + data_seed)

    def batch():
        rows = []
        for _ in range(args.bs):
            line = lines[rng.integers(len(lines))][rng.integers(len(FORMATS))]
            n = args.seq - len(line)
            o = int(rng.integers(0, len(train) - n))
            rows.append(np.concatenate([train[o:o + n].astype(np.int64), np.asarray(line, dtype=np.int64)]))
        return torch.tensor(np.stack(rows), device=dev)

    def geo_loss():
        model(input_ids=num_prefix)
        return sum(1 - cka(store[L][:, -1, :], target) for L in geo_layers) / len(geo_layers)

    @torch.no_grad()
    def evaluate(step) -> dict:
        model.eval()
        res = {"step": step, "tokens": step * args.bs * args.seq}
        with torch.autocast(dev, dtype=torch.bfloat16):
            for name, (ids, ans) in evalsets.items():
                res[name] = float((model(input_ids=ids).logits[:, -1].argmax(-1) == ans).float().mean())
            vr = np.random.default_rng(7)
            vl = []
            for _ in range(8):
                o = vr.integers(0, len(val) - args.seq - 1, size=16)
                x = torch.tensor(np.stack([val[i:i + args.seq + 1].astype(np.int64) for i in o]), device=dev)
                vl.append(float(Fn.cross_entropy(model(input_ids=x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten())))
            res["val_loss"] = float(np.mean(vl))
            model(input_ids=num_prefix)
            res["cka_helix"] = {f"L{L}": float(cka(store[L][:, -1, :], K["helix"])) for L in geo_layers}
        model.train()
        return res

    curve, t0 = [], time.time()
    model.train()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(s)
        x = batch()
        with torch.autocast(dev, dtype=torch.bfloat16):
            lm = model(input_ids=x, labels=x).loss
            gl = geo_loss() if geo_on(s) else torch.zeros((), device=dev)
            loss = lm + args.lam * gl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % args.eval_every == 0 or s == steps - 1:
            r = evaluate(s); r.update(lm=float(lm), geo=float(gl), sec=time.time() - t0); curve.append(r)
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps({"args": vars(args), "n_params": n_params, "steps": steps, "geo_layers": geo_layers,
                                        "curve": curve}, indent=1))
    print(f"done {args.arm} seed{args.seed} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
