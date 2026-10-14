"""36-domain fine-tuning test of the manifold prior (2026-10-14; user request: test every atlas domain with a prior).

For every domain of tools/manifold_domains.py: a supervised task built from the domain's own question family (the
atlas test templates, zero-shot), a held-out split without equivalent leakage, and LoRA fine-tuning of one OLMo-2
checkpoint with
  none        task loss only
  prior       + lam * (1 - CKA(states of "Q: <item>" (last token) at --layers, Gram of the domain's PRIMARY target))
  prior_shuf  the same target with the items permuted, a new permutation per seed (as in the ptu audit), redrawn until
              CKA(shuffled, true) < 0.15 when possible (else the farthest of 2000 draws)
Accuracy = the right answer ranks first among the candidates by summed log-prob of " <answer>" (all items for the
"k steps after x" domains, the two options for comparisons); the train loss is the cross-entropy of those tokens.
Splits (fixed by --split_seed; the same for every arm and seed):
  offset, cyclic   questions "k <unit> after x" with k = 1..2n; hold out 40% of the (item, k mod n) classes with
                   k mod n != 0: a held-out question needs the shift learned on OTHER items (no equivalent leakage)
  offset, other    hold out 40% of the (item, k) questions; every item and every k still appear in training
  comparisons      all comparisons of the domain's rule (manifold_domains._pair_answer); hold out 40% of the unordered
                   option pairs {b, c} (the anchor a, if any, goes with its pair); both orders in the same split
Writes <out>/<domain>/run_<arm>_seed<seed>.json (test / train accuracy, geometry CKA before / after) and base.json.

    python tools/ft36.py --model <ckpt dir> --out <dir> --domains days,countries --seeds 0,1,2
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
from manifold_atlas import PRIMARY, PRIMARY_OVERRIDE, primary
from manifold_domains import OFFSET, PAIR, PAIR2, WORD_NUM, _pair_answer, _singular, domains
from pretrain_helix import centred_gram, cka

CYCLIC_UNIT = {"days": "days", "months": "months", "hours": "hours", "pitch": "semitones", "keys": "semitones",
               "zodiac": "signs", "chinese_zodiac": "years"}


def offset_questions(name, d):
    """[(prompt, answer index, class key)] for an offset domain; cyclic domains use k = 1..2n so every (item, shift)
    class has several surface forms (and the class split is meaningful)."""
    tmpl, mode, ks = OFFSET[name]
    items, n = d["items"], len(d["items"])
    out = []
    if mode == "cyclic":
        ks = range(1, 2 * n + 1)
    for i in range(n):
        for k in ks:
            if mode == "add":
                a = int(d["coord"][i])
                if a + k <= 99:
                    out.append((tmpl.format(x=a, k=k), items.index(str(a + k)), (i, k)))
                continue
            j = (i + k) % n if mode == "cyclic" else i + k
            if j >= n or (name == "number_words" and items[j] not in WORD_NUM):
                continue
            kw = str(k) if (name in ("years", "hours") or k > 20) else list(WORD_NUM)[k - 1]
            key = (i, k % n) if mode == "cyclic" else (i, k)
            out.append((_singular(tmpl, items[i], kw, k), j, key))
    return out, mode == "cyclic"


def pair_questions(name, d, cap=1500, seed=7):
    """[(prompt, (b, c) option items, answer index 0/1, unordered option-pair key)] for a comparison domain: every
    valid comparison of the domain's rule (manifold_domains._pair_answer, same margins as the atlas tests), at most
    `cap` comparisons (random subset), each asked in both orders."""
    items, n = d["items"], len(d["items"])
    tmpl = PAIR2[name][0] if name in PAIR2 else PAIR[name]
    uses_a = "{a}" in tmpl
    comps = []
    for a in (range(n) if uses_a else [0]):
        for b in range(n):
            for e in range(b + 1, n):
                if uses_a and a in (b, e):
                    continue
                ans, q = _pair_answer(name, d, a, b, e)
                if ans is not None:
                    comps.append((a, b, e, ans, q))
    if len(comps) > cap:
        rng = np.random.default_rng(seed)
        comps = [comps[i] for i in sorted(rng.permutation(len(comps))[:cap])]
    out = []
    for a, b, e, ans, q in comps:
        for x, y, aa in ((b, e, ans), (e, b, 1 - ans)):
            out.append((tmpl.format(a=items[a], b=items[x], c=items[y], q=q), (x, y), aa, (b, e)))
    return out


def split(keys, frac, seed):
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    held = {uniq[j] for j in rng.permutation(len(uniq))[:int(round(frac * len(uniq)))]}
    return held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="all")
    ap.add_argument("--arms", default="none,prior,prior_shuf")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16")
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test: a small random OLMo-2 with the given tokenizer")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.tiny:
        from transformers import Olmo2Config, Olmo2ForCausalLM
        model = Olmo2ForCausalLM(Olmo2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                             num_attention_heads=4, num_key_value_heads=4))
        layers_geo = [1, 2]
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        layers_geo = [int(v) for v in args.layers.split(",")]
    model = model.to(dev)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    blocks = model.base_model.model.model.layers
    store = {}
    for L in layers_geo:
        blocks[L - 1].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))
    D = domains()
    names = list(D) if args.domains == "all" else args.domains.split(",")
    arms = args.arms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out)

    def ids_of(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def seq_logprobs(pairs, bs=128):
        """Summed log-prob of each continuation after its prompt (grouped by total length, no padding)."""
        seqs = [(ids_of(p), ids_of(c)) for p, c in pairs]
        score = torch.zeros(len(seqs), device=dev)
        by_len = {}
        for i, (a, b) in enumerate(seqs):
            by_len.setdefault((len(a) + len(b), len(b)), []).append(i)
        for (L, m), idx in by_len.items():
            for s0 in range(0, len(idx), bs):
                b_ = idx[s0:s0 + bs]
                x = torch.tensor([seqs[i][0] + seqs[i][1] for i in b_], device=dev)
                lp = torch.log_softmax(model(input_ids=x, logits_to_keep=m + 1).logits[:, :-1].float(), -1)
                score[b_] = lp.gather(-1, x[:, L - m:, None])[..., 0].sum(1)
        return score

    for name in names:
        d = D[name]
        items = d["items"]
        dom_dir = out_root / name
        dom_dir.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------------ questions and split
        if name in OFFSET:
            qs, cyclic = offset_questions(name, d)
            held = split([q[2] for q in qs], 0.4, args.split_seed)
            if cyclic:
                held = {h for h in held if h[1] != 0}
            else:   # give back held combos whose item or k would otherwise never be trained (e.g. 89 + 10 is 89's only combo)
                for h in sorted(held, key=lambda h: (h[1], h[0])):
                    rest = {k for _, _, k in qs} - held
                    if h[0] not in {k[0] for k in rest} or h[1] not in {k[1] for k in rest}:
                        held.discard(h)
            train = [(p, list(range(len(items))), j) for p, j, k in qs if k not in held]
            test = [(p, list(range(len(items))), j) for p, j, k in qs if k in held]
            tr_keys = {k for _, _, k in qs if k not in held}
            assert not (tr_keys & held)
            if not cyclic:   # every item and every k still trained
                assert {k[0] for k in tr_keys} >= {k[0] for k in held} and {k[1] for k in tr_keys} >= {k[1] for k in held}
            else:
                assert {k[1] for k in tr_keys if k[1]} >= {k[1] for k in held}
        else:
            qs = pair_questions(name, d)
            held = split([q[3] for q in qs], 0.4, args.split_seed)
            train = [(p, list(opts), a) for p, opts, a, k in qs if k not in held]
            test = [(p, list(opts), a) for p, opts, a, k in qs if k in held]
            assert not ({k for _, _, _, k in qs if k not in held} & held)
        train_eval = [train[i] for i in np.random.default_rng(9).permutation(len(train))[:400]]   # train accuracy on <= 400
        print(f"== {name}: {len(train)} train / {len(test)} test questions; e.g. {train[0][0]!r} -> {items[train[0][1][train[0][2]]]!r}", flush=True)
        # eval candidates
        def accuracy(rows):
            pairs, owner = [], []
            for r, (p, opts, a) in enumerate(rows):
                for c, it in enumerate(opts):
                    pairs.append((p, " " + items[it])); owner.append((r, c))
            sc = seq_logprobs(pairs).cpu().numpy()
            best = {}
            for (r, c), v in zip(owner, sc):
                if r not in best or v > best[r][1]:
                    best[r] = (c, v)
            return float(np.mean([best[r][0] == rows[r][2] for r in range(len(rows))]))
        # prior: states of "Q: <item>" (last token), primary target, shuffled target
        pref = [ids_of(f"Q: {it}") for it in items]
        groups = {}
        for i, x in enumerate(pref):
            groups.setdefault(len(x), []).append(i)
        T = d["targets"][primary(d, name)]
        G0 = centred_gram(T, "cpu")

        def far_perm(seed):
            rng_p, best = np.random.default_rng(124 + seed), None
            for _ in range(2000):
                q = rng_p.permutation(len(items))
                Gq = centred_gram(T[q], "cpu")
                c = float((G0 * Gq).sum() / (G0.norm() * Gq.norm()))
                if best is None or c < best[0]:
                    best = (c, q)
                if c < 0.15:
                    break
            return best
        shuf = {seed: far_perm(seed) for seed in seeds}
        K = {"prior": centred_gram(T, dev)}
        K.update({("prior_shuf", seed): centred_gram(T[q], dev) for seed, (_, q) in shuf.items()})

        def item_states():
            Hs = {L: torch.zeros(len(items), 0, device=dev) for L in layers_geo}
            parts = {L: [None] * len(items) for L in layers_geo}
            for Ln, idx in groups.items():
                model(input_ids=torch.tensor([pref[i] for i in idx], device=dev))
                for L in layers_geo:
                    h = store[L][:, -1, :]
                    for j, i in enumerate(idx):
                        parts[L][i] = h[j]
            return {L: torch.stack(parts[L]) for L in layers_geo}

        def geo_loss(Kt):
            Hs = item_states()
            return sum(1 - cka(H, Kt) for H in Hs.values()) / len(Hs)

        @torch.no_grad()
        def geometry():
            Hs = item_states()
            return {f"L{L}": float(cka(H, K["prior"])) for L, H in Hs.items()}

        # training batches: prompt + " answer" with the loss on the answer tokens
        enc_tr = [(ids_of(p), ids_of(" " + items[opts[a]])) for p, opts, a in train]

        def train_batch(idx):
            L = max(len(a) + len(b) for a, b in (enc_tr[i] for i in idx))
            x = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
            lab = torch.full((len(idx), L), -100, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                a, b = enc_tr[i]; s = a + b
                x[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1; lab[r, len(a):len(s)] = torch.tensor(b)
            return x.to(dev), att.to(dev), lab.to(dev)

        def reset(seed):
            torch.manual_seed(seed)
            from peft.tuners.lora import LoraLayer
            for m in model.modules():
                if isinstance(m, LoraLayer):
                    m.reset_lora_parameters("default", True)

        model.eval()
        with torch.no_grad():
            base = {"domain": name, "primary": primary(d, name), "n_train": len(train), "n_test": len(test),
                    "chance": float(np.mean([1 / len(o) for _, o, _ in test])),
                    "shuf_cka": {str(k): v[0] for k, v in shuf.items()},
                    "test": accuracy(test), "train": accuracy(train_eval), "geometry": geometry()}
        (dom_dir / "base.json").write_text(json.dumps(base, indent=1))
        print(f"   base: test {base['test']:.3f} train {base['train']:.3f} chance {base['chance']:.3f} geometry {base['geometry']}", flush=True)
        for seed in seeds:
            for arm in arms:
                path = dom_dir / f"run_{arm}_seed{seed}.json"
                if path.exists():
                    continue
                reset(seed)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
                order_rng = np.random.default_rng(1000 + seed)
                model.train()
                log, step = [], 0
                while step < args.steps:
                    order = order_rng.permutation(len(enc_tr))
                    for i in range(0, len(order), args.bs):
                        if step >= args.steps:
                            break
                        x, att, lab = train_batch(order[i:i + args.bs])
                        out_m = model(input_ids=x, attention_mask=att)
                        ce = Fn.cross_entropy(out_m.logits[:, :-1].float().flatten(0, 1), lab[:, 1:].flatten(), ignore_index=-100)
                        loss = ce
                        if arm != "none":
                            gl = geo_loss(K["prior"] if arm == "prior" else K[("prior_shuf", seed)])
                            loss = ce + args.lam * gl
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        if step % 70 == 0:
                            log.append({"step": step, "ce": float(ce.detach()), "geo": float(gl.detach()) if arm != "none" else None})
                        step += 1
                del opt
                model.eval()
                with torch.no_grad():
                    res = {"domain": name, "arm": arm, "seed": seed, "lam": args.lam, "steps": args.steps, "layers": layers_geo,
                           "shuf_cka": shuf[seed][0] if arm == "prior_shuf" else None,
                           "model": args.model, "n_train": len(train), "n_test": len(test), "chance": base["chance"],
                           "test": accuracy(test), "train": accuracy(train_eval), "geometry": geometry(), "log": log}
                path.write_text(json.dumps(res, indent=1))
                print(f"   {arm:10s} seed {seed}: test {res['test']:.3f} train {res['train']:.3f} geometry {res['geometry']} "
                      f"({time.time() - t_start:.0f}s)", flush=True)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
