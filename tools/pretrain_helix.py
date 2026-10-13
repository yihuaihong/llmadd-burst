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
The *_ctx accuracies put the same prompts where training puts them: after a held-out FineWeb chunk, so the prompt ends
at the same position as in a training sequence (the plain ones start the prompt at position 0, which training never
shows; in-batch accuracy reaches ~100% while the plain held-out accuracy stays far lower).

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


def build_model(args, tok, dev):
    from transformers import Olmo2Config, Olmo2ForCausalLM
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=args.d_model, intermediate_size=args.d_ff,
                      num_hidden_layers=args.n_layers, num_attention_heads=args.n_heads, num_key_value_heads=args.n_heads,
                      max_position_embeddings=args.seq, tie_word_embeddings=True,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return Olmo2ForCausalLM(cfg).to(dev)


def arithmetic_sets(args, tok, val, dev):
    """Held-out split and eval sets. test = 1000 ordered pairs never trained on; strict = the test pairs whose swap
    (b, a) is not trained on either (otherwise a + b can be recalled from b + a: 870 of the 1000 test pairs);
    *_ctx = the same prompts after a held-out FineWeb chunk, ending at the index where training prompts end."""
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    pairs = pairs[np.random.default_rng(0).permutation(len(pairs))]
    test, trainp = pairs[:1000], pairs[1000:]
    tr = set(map(tuple, trainp.tolist()))
    assert not (set(map(tuple, test.tolist())) & tr), "held-out pairs leak into training"
    strict = np.array([(b, a) not in tr for a, b in test.tolist()])
    lines = [[tok(f.format(a=a, b=b) + f"{a + b}\n", add_special_tokens=False)["input_ids"] for f in FORMATS] for a, b in trainp]
    for f in range(len(FORMATS)):   # every line ends "... <answer> \n" with the answer as one token
        _l = lines[0][f]
        assert tok.convert_ids_to_tokens(_l[-2:])[0] == str(trainp[0][0] + trainp[0][1]), tok.convert_ids_to_tokens(_l[-3:])
    enc = lambda ps: nl.encode(tok, ps, dev)
    ans_tok = lambda P: torch.tensor([tok(str(a + b), add_special_tokens=False)["input_ids"][0] for a, b in P], device=dev)
    evalsets = {"test": (enc([FORMATS[0].format(a=a, b=b) for a, b in test]), ans_tok(test)),
                "test_plus": (enc([FORMATS[1].format(a=a, b=b) for a, b in test]), ans_tok(test)),
                "train": (enc([FORMATS[0].format(a=a, b=b) for a, b in trainp[:500]]), ans_tok(trainp[:500])),
                "strict": (enc([FORMATS[0].format(a=a, b=b) for a, b in test[strict]]), ans_tok(test[strict]))}
    # the prompt's last token must be the token that precedes the answer in a training line of the same format
    assert int(evalsets["test"][0][0, -1]) == lines[0][0][-3], "eval prompt does not end like a training prompt"
    if args.ctx_eval:
        cr = np.random.default_rng(11)
        def in_context(ids):   # held-out text in front, so the prompt ends where it ends in training (answer at -2)
            n = args.seq - ids.shape[1] - 2
            o = cr.integers(0, len(val) - n, size=ids.shape[0])
            return torch.cat([torch.tensor(np.stack([val[i:i + n].astype(np.int64) for i in o]), device=dev), ids], 1)
        evalsets.update({f"{k}_ctx": (in_context(ids), ans) for k, (ids, ans) in list(evalsets.items())})
        assert all(evalsets[f"{k}_ctx"][0].shape[1] == args.seq - 2 for k in ("test", "train", "strict"))
    return test, trainp, lines, evalsets


def show_inputs(args, tok, train, lines, evalsets, k=3):
    """Decoded training rows next to decoded eval rows; the scored / predicted token is marked with brackets."""
    rng = np.random.default_rng(0)
    def tail(ids, mark):
        ids = [int(t) for t in ids]
        pieces = [tok.decode([t]) for t in ids[-28:]]
        pieces[mark] = "\u27e6" + pieces[mark] + "\u27e7"
        return "".join(pieces).replace("\n", "\\n")
    print("TRAIN rows (answer token marked; scored by the logits one position earlier):")
    for _ in range(k):
        line = sum((lines[rng.integers(len(lines))][rng.integers(len(FORMATS))] for _ in range(args.arith_lines)), [])
        n = args.seq - len(line)
        row = np.concatenate([train[:n].astype(np.int64), np.asarray(line, dtype=np.int64)])
        print(f"  len {len(row)}  answer index {len(row) - 2}: ...{tail(row, -2)}")
    for name in ("test", "test_ctx", "strict_ctx"):
        if name not in evalsets:
            continue
        ids, ans = evalsets[name]
        print(f"EVAL {name} rows (last token marked = scored position; expected answer after it):")
        for i in range(k):
            print(f"  len {ids.shape[1]}  scored index {ids.shape[1] - 1}: ...{tail(ids[i].tolist(), -1)}  -> {tok.decode([int(ans[i])])}")
    print({k: tuple(v[0].shape) for k, v in evalsets.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=("none", "helix", "helix_shuf", "helix_window", "helix_acc", "helix_cka"), required=True)
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
    ap.add_argument("--arith_lines", type=int, default=1, help="arithmetic lines appended per sequence (1 ~ 2%% of tokens, 4 ~ 7%%)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--gate_start", type=float, default=0.1, help="helix_acc / helix_cka: first step of the prior (fraction)")
    ap.add_argument("--gate_thr", type=float, default=0.1, help="helix_acc: release once the EMA of train-batch answer accuracy reaches this")
    ap.add_argument("--cka_release", type=float, default=0.01, help="helix_cka: release once 1 - CKA stays below this ...")
    ap.add_argument("--cka_patience", type=int, default=200, help="... for this many consecutive steps")
    ap.add_argument("--lam_after", type=float, default=0.0, help="helix_window: weight after the window ends (0 = release)")
    ap.add_argument("--geo_layers", default=None, help="default: n_layers/3 and 2 n_layers/3")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ctx_eval", action="store_true", help="also evaluate every prompt in training position (*_ctx keys)")
    ap.add_argument("--save_model", action="store_true", help="save the final weights next to the run file")
    ap.add_argument("--dry_inputs", action="store_true", help="print decoded training and eval inputs side by side, then exit")
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
    test, trainp, lines, evalsets = arithmetic_sets(args, tok, val, dev)
    enc = lambda ps: nl.encode(tok, ps, dev)
    if args.dry_inputs:
        show_inputs(args, tok, train, lines, evalsets)
        return
    num_prefix = enc([f"Q: {x}" for x in range(10, 100)])
    xs = np.arange(10, 100)
    K = {"helix": centred_gram(nl.features(xs)[0], dev),
         "helix_shuf": centred_gram(nl.features(xs[np.random.default_rng(124).permutation(90)])[0], dev)}

    # ------------------------------------------------------------ model
    model = build_model(args, tok, dev)
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
    dyn = {"released": None, "ema": 0.0, "low": 0}

    def geo_weight(s):
        if args.arm == "none":
            return 0.0
        if args.arm in ("helix_acc", "helix_cka"):
            return args.lam if (s >= args.gate_start * steps and dyn["released"] is None) else 0.0
        if args.arm == "helix_window":
            return args.lam if w0 * steps <= s < w1 * steps else (args.lam_after if s >= w1 * steps else 0.0)
        return args.lam
    target = K["helix_shuf" if args.arm == "helix_shuf" else "helix"]
    rng = np.random.default_rng(1000 + data_seed)

    def batch():
        rows = []
        for _ in range(args.bs):
            line = sum((lines[rng.integers(len(lines))][rng.integers(len(FORMATS))] for _ in range(args.arith_lines)), [])
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
                hit = [model(input_ids=ids[i:i + 250], logits_to_keep=1).logits[:, -1].argmax(-1) == ans[i:i + 250]
                       for i in range(0, len(ids), 250)]
                res[name] = float(torch.cat(hit).float().mean())
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
            out_m = model(input_ids=x, labels=x)
            lm = out_m.loss
            gw = geo_weight(s)
            gl = geo_loss() if gw > 0 else torch.zeros((), device=dev)
            loss = lm + gw * gl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            bacc = float((out_m.logits[:, -3].argmax(-1) == x[:, -2]).float().mean())
        dyn["ema"] = 0.98 * dyn["ema"] + 0.02 * bacc
        if dyn["released"] is None and gw > 0:
            if args.arm == "helix_acc" and dyn["ema"] >= args.gate_thr:
                dyn["released"] = s
            if args.arm == "helix_cka":
                dyn["low"] = dyn["low"] + 1 if float(gl) < args.cka_release else 0
                if dyn["low"] >= args.cka_patience:
                    dyn["released"] = s
        if s % args.eval_every == 0 or s == steps - 1:
            r = evaluate(s); r.update(lm=float(lm), geo=float(gl), sec=time.time() - t0, ema_train_acc=dyn["ema"],
                                      released=dyn["released"]); curve.append(r)
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps({"args": vars(args), "n_params": n_params, "steps": steps, "geo_layers": geo_layers,
                                        "curve": curve}, indent=1))
    if args.save_model:
        torch.save(model.state_dict(), out / f"model_{args.arm}_seed{args.seed}{args.tag}.pt")
    print(f"done {args.arm} seed{args.seed} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
