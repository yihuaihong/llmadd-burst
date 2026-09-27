"""Full fine-tuning with the geometry (CKA) loss on several GPUs (FSDP), same protocol as manifold_train.py --mode full.

fp32 master weights sharded across ranks, bf16 compute. --optim adamw (standard; 2 x 80 GB) or adamw8bit
(bitsandbytes; needed on 2 x 40 GB). Data, geometries, readouts, evaluation and output files are the ones
of manifold_train.py, so results are comparable run for run (up to the optimiser when adamw8bit is used).

    torchrun --nproc_per_node=2 tools/manifold_train_fsdp.py --model <dir> --out <dir> --optim adamw
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as Fn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numlib as nl
from manifold_train import GEOMS, cka, decoder_layers, diagnostics, geom_features, summarize, target_grams, terse_prompt, three_prompt, two_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--optim", choices=("adamw", "adamw8bit"), default="adamw")
    ap.add_argument("--geoms", default=",".join(GEOMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--train_pairs", type=int, default=1500)
    ap.add_argument("--test_pairs", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16, help="global batch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_bs", type=int, default=128, help="per rank")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    geoms = args.geoms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    man_layers = [int(v) for v in args.layers.split(",")]
    t0 = time.time()
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    layer_cls = type(decoder_layers(model)[0])
    model = FSDP(model,
                 auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}),
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32) if cuda else None,
                 sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=dev if cuda else None, use_orig_params=True)
    layers = decoder_layers(model)
    d = model.config.hidden_size if hasattr(model, "config") else model.module.config.hidden_size
    init = [p.detach().clone().cpu() for p in model.parameters()]          # this rank's shard of the base weights

    num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
    toks = tok.convert_ids_to_tokens(tok(two_prompt(23, 45))["input_ids"])
    assert toks[-6:] == ["23", "Ġ+", "Ġ", "45", "Ġ=", "Ġ"], toks
    slot_pos = {"A": len(toks) - 6, "B": len(toks) - 3}

    store: dict = {}
    for L in man_layers:
        layers[L].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))

    def forward(ids):
        return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()

    # ---------------------------------------------------------------- data (identical to manifold_train.py)
    rng = np.random.default_rng(0)
    pairs = np.array([(a, b) for a in range(10, 100) for b in range(10, 100)])
    pairs = pairs[rng.permutation(len(pairs))]
    train, test = pairs[: args.train_pairs], pairs[args.train_pairs: args.train_pairs + args.test_pairs]
    for j in range(2):
        assert set(np.unique(train[:, j])) >= set(range(10, 100))
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

    def my_chunks(n: int, bs: int):
        """Equal-length per-rank index chunks (padded by repetition; `real` masks the padding) - every rank
        must run the same number of forwards under FSDP."""
        per = -(-n // world)
        idx = np.arange(rank * per, (rank + 1) * per)
        real = idx < n
        idx = np.minimum(idx, n - 1)
        for i in range(0, per, bs):
            yield idx[i:i + bs], real[i:i + bs]

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        res = {}
        for name, (ids, ans) in evalsets.items():
            good = torch.zeros(2, device=dev)
            for idx, real in my_chunks(len(ids), args.eval_bs):
                am = forward(ids[torch.tensor(idx, device=dev)]).argmax(-1)
                ok = (am == num_ids[torch.tensor(ans[idx], device=dev)]) & torch.tensor(real, device=dev)
                good += torch.stack([ok.sum().float(), torch.tensor(float(real.sum()), device=dev)])
            dist.all_reduce(good)
            res[name] = float(good[0] / good[1])
        return res

    @torch.no_grad()
    def number_means() -> dict:
        model.eval()
        sums = {(L, s): torch.zeros(90, d, device=dev) for L in man_layers for s in slot_pos}
        cnt = {s: torch.zeros(90, device=dev) for s in slot_pos}
        for idx, real in my_chunks(len(train), args.eval_bs):
            forward(train_ids[torch.tensor(idx, device=dev)])
            w = torch.tensor(real, device=dev, dtype=torch.float32)
            for j, s in enumerate(slot_pos):
                v = torch.tensor(train[idx, j] - 10, device=dev)
                for L in man_layers:
                    sums[(L, s)].index_add_(0, v, store[L][:, slot_pos[s], :].float() * w[:, None])
                cnt[s].index_add_(0, v, w)
        for t in list(sums.values()) + list(cnt.values()):
            dist.all_reduce(t)
        return {k: (v / cnt[k[1]][:, None]).cpu().numpy() for k, v in sums.items()}

    grams = target_grams(perm, dev)
    num_prefix = enc([f"Q: {x}" for x in range(10, 100)])   # every rank computes the full 90-number geometry
    assert num_prefix.shape[1] == slot_pos["A"] + 1

    def geo_states():
        forward(num_prefix)
        return {L: store[L][:, slot_pos["A"], :] for L in man_layers}

    @torch.no_grad()
    def geometry_report() -> dict:
        model.eval()
        Hs = geo_states()
        return {f"L{L}": {"cka_" + G: float(cka(Hs[L], grams[G])) for G in grams} for L in man_layers}

    base_means = number_means()
    base_eval = evaluate()
    base_geo = geometry_report()
    if rank == 0:
        (out / "base.json").write_text(json.dumps({"eval": base_eval, "manifold": diagnostics(base_means, None, slot_pos),
                                                   "geometry": base_geo}, indent=1))
    say(f"base: {base_eval} ({time.time() - t0:.0f}s)")

    for seed in seeds:
        for G in geoms:
            path = out / f"run_{G}_seed{seed}.json"
            if path.exists():
                continue
            with torch.no_grad():
                for p, s in zip(model.parameters(), init):
                    p.copy_(s.to(p.device))
            torch.manual_seed(seed)
            params = [p for p in model.parameters() if p.requires_grad]
            if args.optim == "adamw8bit":
                import bitsandbytes as bnb
                opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
            else:
                opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
            order_rng = np.random.default_rng(1000 + seed)
            log = []
            model.train()
            for ep in range(args.epochs):
                order = order_rng.permutation(len(train))
                ce_s = man_s = 0.0; nb = 0
                for i in range(0, len(order), args.bs):
                    bi = order[i:i + args.bs][rank::world]
                    chunk = train[bi]
                    logits = forward(train_ids[torch.tensor(bi, device=dev)])
                    ce = Fn.cross_entropy(logits, num_ids[torch.tensor(chunk.sum(1), device=dev)])
                    if G == "none":
                        with torch.no_grad():
                            Hs = geo_states()
                            ml = sum(1 - cka(Hs[L], grams["helix"]) for L in man_layers) / len(man_layers)
                        loss = ce
                    else:
                        Hs = geo_states()
                        ml = sum(1 - cka(Hs[L], grams[G]) for L in man_layers) / len(man_layers)
                        loss = ce + args.lam * ml
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    ce_s += float(ce); man_s += float(ml); nb += 1
                stats = torch.tensor([ce_s / nb, man_s / nb], device=dev); dist.all_reduce(stats); stats /= world
                log.append({"epoch": ep, "ce": float(stats[0]), "geo_loss": float(stats[1])})
            del opt
            ev = evaluate()
            means = number_means()
            geo = geometry_report()
            if rank == 0:
                path.write_text(json.dumps({"mode": f"full_fsdp_{args.optim}", "loss": "v2_cka", "geom": G, "seed": seed, "lam": args.lam,
                                            "lr": args.lr, "layers": man_layers, "world": world, "train_log": log, "eval": ev,
                                            "manifold": diagnostics(means, None, slot_pos), "geometry": geo}, indent=1))
            mem = torch.tensor([torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0], device=dev)
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            say(f"fsdp {G} seed{seed}: {ev} ({time.time() - t0:.0f}s, peak GPU mem {float(mem):.1f} GB/rank)")
            dist.barrier()
    if rank == 0:
        summarize(out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
