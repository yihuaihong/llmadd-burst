"""36-domain fine-tuning test of the manifold prior (2026-10-14; user request: test every atlas domain with a prior).

For every domain of tools/manifold_domains.py: a supervised task built from the domain's own question family (the atlas
test templates, prefixed with "Q: "), a held-out split without equivalent or commutative leakage, and LoRA fine-tuning of
one OLMo-2 checkpoint with
  none        task loss only
  prior       + lam * (1 - CKA(states of the domain's items INSIDE the task prompts, at --layers; Gram of the domain's
              primary target)), on a fresh batch of --geo_bs training prompts every step. These are the states the task
              reads (F71: shape the computation site; in F10 / F40 the "Q: <item>" probe and the task prompt shared them)
  prior_shuf  the same with the target rows permuted, a new far permutation per seed (CKA(shuffled, true) < 0.15 when
              possible, else the farthest of 2000 draws)
Accuracy = the right answer ranks first among the candidates by summed log-prob of " <answer>\\n" (the newline closes the
answer, so " C" cannot win over " C#" by being its prefix); candidates = all items for "k steps after x", the two
options for comparisons. The training loss is the cross-entropy of the same tokens.
Splits (per seed: --split_seed -1 = the seed itself; the same for every arm of a seed):
  offset, cyclic   "k <unit> after x" with k = 1..2n; hold out 40% of the (item, k mod n) classes with k mod n != 0
  offset, other    hold out 40% of the question keys; numbers and number_words use the UNORDERED key {x, k} so that
                   x + k and k + x are held out together; keys are given back until every x and every k still occur in
                   training
  comparisons      hold out 40% of the unordered option pairs {b, c} (both orders, every anchor)
Base = the checkpoint with the adapter disabled. Targets: columns z-scored over the domain's items as in the atlas,
except "tree" / "within" targets, whose column weights encode the structure (raw centred features).
Audit 2026-10-14 (runs/ft36/AUDIT.md) changed: the prior site (was "Q: <item>" probes, a state no task prompt contains),
the answer terminator, the commutative split, base with the adapter disabled (the base of a job's 2nd..6th domain had
been measured with the previous domain's adapter), raw tree / within Grams, per-seed splits, hooks on layers[L] as in
F10 / F40, atomic result files.
Writes <out>/<domain>/base_seed<s>.json and run_<arm>_seed<s>.json (test / train accuracy, per-question test hits,
geometry of the item states in held-out prompts and of context-free "Q: <item>" probes).

    python tools/ft36.py --model <ckpt dir> --out <dir> --domains days,countries --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import string
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import numlib as nl
from manifold_atlas import primary
from manifold_domains import OFFSET, PAIR, PAIR2, WORD_NUM, _pair_answer, _singular, domains
from pretrain_helix import cka

RAW_GRAM = ("tree", "within")   # target names whose column weights carry the structure: no per-column z-scoring
PREFIX = "Q: "


def fill(tmpl, **vals):
    """Format tmpl and return (text, {field: (start, end) character span})."""
    out, spans = "", {}
    for lit, field, _, _ in string.Formatter().parse(tmpl):
        out += lit
        if field is not None:
            v = str(vals[field])
            spans[field] = (len(out), len(out) + len(v))
            out += v
    return out, spans


def sing_tmpl(tmpl, k):
    """The template with the unit word after {k} made singular for k = 1 (manifold_domains._singular on the template)."""
    w = tmpl.split(" ")
    at = w.index("{k}") + 1
    if k == 1 and at < len(w) and w[at].endswith("s") and w[at] != "is":
        w[at] = w[at][:-1]
    return " ".join(w)


def shift(span, n=len(PREFIX)):
    return span[0] + n, span[1] + n


def offset_questions(name, d):
    """[dict(prompt, occ = [(item, char span)], ans, key, xk = (x id, k))] for an offset domain, and whether it is
    cyclic. Cyclic domains use k = 1..2n so every (item, shift) class has several surface forms."""
    tmpl, mode, ks = OFFSET[name]
    items, n = d["items"], len(d["items"])
    out = []
    if mode == "cyclic":
        ks = range(1, 2 * n + 1)
    for i in range(n):
        for k in ks:
            if mode == "add":
                a = int(d["coord"][i])
                if a + k > 99:
                    continue
                text, sp = fill(tmpl, x=a, k=k)
                out.append(dict(prompt=PREFIX + text, occ=[(i, shift(sp["x"]))], ans=items.index(str(a + k)),
                                key=(min(a, k), max(a, k)), xk=(a, k)))
                continue
            j = (i + k) % n if mode == "cyclic" else i + k
            if j >= n or (name == "number_words" and items[j] not in WORD_NUM):
                continue
            kw = str(k) if (name in ("years", "hours") or k > 20) else list(WORD_NUM)[k - 1]
            text, sp = fill(sing_tmpl(tmpl, k), x=items[i], k=kw)
            assert text == _singular(tmpl, items[i], kw, k), (text, _singular(tmpl, items[i], kw, k))
            if mode == "cyclic":
                key = (i, k % n)
            elif name == "number_words":   # "three plus five" and "five plus three" together
                v = WORD_NUM[items[i]]
                key = (min(v, k), max(v, k))
            else:
                key = (i, k)
            out.append(dict(prompt=PREFIX + text, occ=[(i, shift(sp["x"]))], ans=j, key=key,
                            xk=(WORD_NUM[items[i]] if name == "number_words" else i, k)))
    return out, mode == "cyclic"


def pair_questions(name, d, cap=1500, seed=7):
    """Every valid comparison of the domain's rule (manifold_domains._pair_answer, the atlas margins), at most `cap`
    (random subset), each asked in both orders; occ = the anchor (if any) and both options."""
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
            text, sp = fill(tmpl, b=items[x], c=items[y], q=q, a=items[a])
            occ = [(x, shift(sp["b"])), (y, shift(sp["c"]))] + ([(a, shift(sp["a"]))] if uses_a else [])
            out.append(dict(prompt=PREFIX + text, occ=occ, ans=aa, opts=(x, y), key=(b, e)))
    return out


def split(keys, frac, seed):
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    return {uniq[j] for j in rng.permutation(len(uniq))[:int(round(frac * len(uniq)))]}


def dump(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def done(path):
    try:
        return "test" in json.loads(path.read_text())
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="all")
    ap.add_argument("--arms", default="none,prior,prior_shuf")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8,12,16", help="hooks on model.layers[L] (output of block L+1), as F10 / F40")
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--geo_bs", type=int, default=64, help="training prompts per step for the prior (fresh each step)")
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--split_seed", type=int, default=-1, help="-1: the split follows the seed")
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
        blocks[L].register_forward_hook(lambda _m, _i, o, L=L: store.__setitem__(L, nl._hidden(o)))
    D = domains()
    names = list(D) if args.domains == "all" else args.domains.split(",")
    arms = args.arms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def ids_of(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def last_token(prompt, span):
        """Index of the token that holds the last character of `span` in the tokenised prompt."""
        offs = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        hit = [t for t, (s, e) in enumerate(offs) if s <= span[1] - 1 < e]   # 2 byte tokens can share one character
        assert hit, (prompt, span, offs)
        return hit[-1]

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

    def padded(rows):
        L = max(len(r) for r in rows)
        x = torch.full((len(rows), L), pad_id, dtype=torch.long); att = torch.zeros((len(rows), L), dtype=torch.long)
        for r, s in enumerate(rows):
            x[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
        return x.to(dev), att.to(dev)

    for name in names:
        d = D[name]
        items = d["items"]
        dom_dir = out_root / name
        dom_dir.mkdir(parents=True, exist_ok=True)
        tname = primary(d, name)
        T = np.asarray(d["targets"][tname], dtype=np.float64)
        T = T[:, T.std(0) > 1e-9]
        Fz = (T - T.mean(0)) if tname in RAW_GRAM else (T - T.mean(0)) / T.std(0)   # fixed over all items

        def gram(F):
            F = torch.tensor(F - F.mean(0), dtype=torch.float32, device=dev)
            return F @ F.T

        def far_perm(seed):
            G0 = gram(Fz)
            rng_p, best = np.random.default_rng(124 + seed), None
            for _ in range(2000):
                q = rng_p.permutation(len(items))
                Gq = gram(Fz[q])
                c = float((G0 * Gq).sum() / (G0.norm() * Gq.norm()))
                if best is None or c < best[0]:
                    best = (c, q)
                if c < 0.15:
                    break
            return best

        # ------------------------------------------------------------ questions (prompt token positions of the items)
        if name in OFFSET:
            qs, cyclic = offset_questions(name, d)
        else:
            qs, cyclic = pair_questions(name, d), None
        for q in qs:
            q["ids"] = ids_of(q["prompt"])
            q["pos"] = [(it, last_token(q["prompt"], sp)) for it, sp in q["occ"]]
            for it, t in q["pos"]:   # the prompt decoded up to that token ends with the item
                head = tok.decode(q["ids"][:t + 1])
                assert items[it] in head[-(len(items[it]) + 4):], (q["prompt"], items[it], head[-40:])
        cand = [" " + it + "\n" for it in items]
        ctoks = [tuple(ids_of(c)) for c in cand]
        assert not any(a != b and b[:len(a)] == a for a in ctoks for b in ctoks), "a candidate is a prefix of another"

        def make_split(sseed):
            keys = [q["key"] for q in qs]
            held = split(keys, 0.4, sseed)
            if cyclic:
                held = {h for h in held if h[1] != 0}
            elif cyclic is False:   # give back keys until every x and every k still occur in training
                by_key = {}
                for q in qs:
                    by_key.setdefault(q["key"], []).append(q)
                while True:
                    tr = [q for q in qs if q["key"] not in held]
                    xs, ks_ = {q["xk"][0] for q in tr}, {q["xk"][1] for q in tr}
                    bad = [h for h in sorted(held) if any(q["xk"][0] not in xs or q["xk"][1] not in ks_ for q in by_key[h])]
                    if not bad:
                        break
                    held.discard(bad[0])
            train = [q for q in qs if q["key"] not in held]
            test = [q for q in qs if q["key"] in held]
            assert not ({q["key"] for q in train} & {q["key"] for q in test})
            if cyclic:
                assert {q["key"][1] for q in train if q["key"][1]} >= {q["key"][1] for q in test}
            elif cyclic is False:
                trxk = {q["xk"] for q in train}
                assert {q["xk"][0] for q in train} >= {q["xk"][0] for q in test}
                assert {q["xk"][1] for q in train} >= {q["xk"][1] for q in test}
                if name in ("numbers", "number_words"):   # no held-out problem has its swap in training
                    assert not any((q["xk"][1], q["xk"][0]) in trxk for q in test)
            return train, test

        def cands(q):
            return list(range(len(items))) if name in OFFSET else list(q["opts"])

        def right(q):
            return q["ans"] if name in OFFSET else q["opts"][q["ans"]]

        def accuracy(rows):
            pairs, owner = [], []
            for r, q in enumerate(rows):
                for it in cands(q):
                    pairs.append((q["prompt"], cand[it])); owner.append((r, it))
            sc = seq_logprobs(pairs).cpu().numpy()
            best = {}
            for (r, it), v in zip(owner, sc):
                if r not in best or v > best[r][1]:
                    best[r] = (it, v)
            return [int(best[r][0] == right(q)) for r, q in enumerate(rows)]

        def site_states(rows):
            """States at the items' last tokens inside the prompts of `rows`: {L: [m, d]} and the item ids [m]."""
            x, att = padded([q["ids"] for q in rows])
            model(input_ids=x, attention_mask=att)
            rr = torch.tensor([r for r, q in enumerate(rows) for _ in q["pos"]], device=dev)
            pp = torch.tensor([t for q in rows for _, t in q["pos"]], device=dev)
            its = np.array([it for q in rows for it, _ in q["pos"]])
            return {L: store[L][rr, pp] for L in layers_geo}, its

        def prior_loss(rows, F):
            H, its = site_states(rows)
            if len(set(its.tolist())) < 3:
                return None
            K = gram(F[its])
            return sum(1 - cka(h, K) for h in H.values()) / len(H)

        pref = [ids_of(PREFIX + it) for it in items]

        @torch.no_grad()
        def geometry(test):
            sel = [test[i] for i in np.random.default_rng(5).permutation(len(test))[:256]]
            Hs, its = {L: [] for L in layers_geo}, []
            for i in range(0, len(sel), 64):
                H, it = site_states(sel[i:i + 64])
                for L in layers_geo:
                    Hs[L].append(H[L])
                its.append(it)
            its = np.concatenate(its)
            K = gram(Fz[its])
            ctx = {f"L{L}": float(cka(torch.cat(Hs[L]), K)) for L in layers_geo}
            probe = {L: [None] * len(items) for L in layers_geo}
            groups = {}
            for i, x in enumerate(pref):
                groups.setdefault(len(x), []).append(i)
            for _, idx in groups.items():
                model(input_ids=torch.tensor([pref[i] for i in idx], device=dev))
                for L in layers_geo:
                    for j, i in enumerate(idx):
                        probe[L][i] = store[L][j, -1, :]
            Kp = gram(Fz)
            return {"ctx": ctx, "probe": {f"L{L}": float(cka(torch.stack(probe[L]), Kp)) for L in layers_geo}}

        def train_batch(rows):
            enc = [(q["ids"], ids_of(cand[right(q)])) for q in rows]
            L = max(len(a) + len(b) for a, b in enc)
            x = torch.full((len(rows), L), pad_id, dtype=torch.long)
            lab = torch.full((len(rows), L), -100, dtype=torch.long); att = torch.zeros((len(rows), L), dtype=torch.long)
            for r, (a, b) in enumerate(enc):
                s = a + b
                x[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1; lab[r, len(a):len(s)] = torch.tensor(b)
            return x.to(dev), att.to(dev), lab.to(dev)

        def reset(seed):
            torch.manual_seed(seed)
            from peft.tuners.lora import LoraLayer
            for m in model.modules():
                if isinstance(m, LoraLayer):
                    m.reset_lora_parameters("default", True)

        for seed in seeds:
            sseed = seed if args.split_seed < 0 else args.split_seed
            train, test = make_split(sseed)
            train_eval = [train[i] for i in np.random.default_rng(9).permutation(len(train))[:400]]
            ans_trained = {right(q) for q in train}
            info = {"domain": name, "primary": tname, "raw_gram": tname in RAW_GRAM, "split_seed": sseed,
                    "n_train": len(train), "n_test": len(test),
                    "n_test_answer_trained": int(sum(right(q) in ans_trained for q in test)),
                    "chance": float(np.mean([1 / len(cands(q)) for q in test])), "layers": layers_geo, "model": args.model}
            shuf_cka, perm = far_perm(seed)
            if seed == seeds[0]:
                q0 = train[0]
                print(f"== {name}: {len(train)} train / {len(test)} test; e.g. {q0['prompt']!r} -> {cand[right(q0)]!r}; item tokens "
                      f"{[tok.decode([q0['ids'][t]]) for _, t in q0['pos']]}", flush=True)
            bpath = dom_dir / f"base_seed{seed}.json"
            if not done(bpath):
                model.eval()
                with torch.no_grad(), model.disable_adapter():
                    hits = accuracy(test)
                    base = dict(info, test=float(np.mean(hits)), test_hits=hits, train=float(np.mean(accuracy(train_eval))),
                                geometry=geometry(test))
                dump(bpath, base)
                print(f"   base seed {seed}: test {base['test']:.3f} train {base['train']:.3f} chance {info['chance']:.3f} "
                      f"geometry {base['geometry']}", flush=True)
            for arm in arms:
                path = dom_dir / f"run_{arm}_seed{seed}.json"
                if done(path):
                    continue
                reset(seed)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
                order_rng, geo_rng = np.random.default_rng(1000 + seed), np.random.default_rng(2000 + seed)
                F = Fz if arm == "prior" else Fz[perm]
                model.train()
                log, step, skipped = [], 0, 0
                while step < args.steps:
                    order = order_rng.permutation(len(train))
                    for i in range(0, len(order), args.bs):
                        if step >= args.steps:
                            break
                        x, att, lab = train_batch([train[j] for j in order[i:i + args.bs]])
                        out_m = model(input_ids=x, attention_mask=att)
                        ce = Fn.cross_entropy(out_m.logits[:, :-1].float().flatten(0, 1), lab[:, 1:].flatten(), ignore_index=-100)
                        loss, gl = ce, None
                        if arm != "none":
                            gl = prior_loss([train[j] for j in geo_rng.integers(0, len(train), args.geo_bs)], F)
                            if gl is None:
                                skipped += 1
                            else:
                                loss = ce + args.lam * gl
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        if step % 70 == 0:
                            log.append({"step": step, "ce": float(ce.detach()), "geo": float(gl.detach()) if gl is not None else None})
                        step += 1
                del opt
                model.eval()
                with torch.no_grad():
                    hits = accuracy(test)
                    res = dict(info, arm=arm, seed=seed, lam=args.lam, steps=args.steps, geo_bs=args.geo_bs,
                               shuf_cka=shuf_cka if arm == "prior_shuf" else None, prior_skipped=skipped,
                               test=float(np.mean(hits)), test_hits=hits, train=float(np.mean(accuracy(train_eval))),
                               geometry=geometry(test), log=log)
                dump(path, res)
                print(f"   {arm:10s} seed {seed}: test {res['test']:.3f} train {res['train']:.3f} geometry {res['geometry']} "
                      f"({time.time() - t_start:.0f}s)", flush=True)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
