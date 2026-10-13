"""Manifold atlas: for every concept domain in manifold_domains.py and one checkpoint, how well each candidate target
geometry is decodable from the item states, and how much the domains' subspaces overlap. No training.

States: the last token of each prompt ("neutral" = "Q: {item}", "cued" = the domain's own template), at every 4th
layer, z-scored per dimension over all items of all domains (removes the outlier dimensions), centred per domain.
  fit      leave-one-out R^2 of a kernel ridge map from the states to the target features (alpha chosen by LOO from a
           grid, separately for the true and for every permuted assignment), averaged over the target's columns;
           null = 200 permutations of the item -> target assignment. Also linear CKA(states, target) with its null.
  basis    per domain, the PLS directions of its primary target (orthonormalised H^T T), i.e. the subspace that
           carries the geometry.
  overlap  ||B_a^T B_b||_F^2 / min(dim a, dim b) for every pair, and the same for bases fitted to permuted targets
           (chance level for these anisotropic states).
  removal  LOO R^2 of domain b after projecting domain a's basis out of b's states: how much b's geometry lives in a's
           subspace (asymmetric; the diagonal is the full removal).
  tasks    3-shot accuracy on each domain's test set (manifold_domains.tasks: "k steps after x" over the domain's
           items, or a two-way comparison that needs the geometry), answer chosen by summed log-prob among candidates.
Writes <out>/atlas_<tag>.json and the states of layers 8/16/24 (fp16) as <out>/states_<tag>.npz.

    python tools/manifold_atlas.py --model <dir> --tag 9B --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from manifold_domains import domains, tasks

PRIMARY = {"helix": "helix", "cyclic": "fourier", "linear": "curve", "sphere": "harmonics", "plane": "plane"}
LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32)
N_NULL = 200


def primary(d):
    return PRIMARY[d["kind"]] if PRIMARY[d["kind"]] in d["targets"] else next(iter(d["targets"]))


@torch.no_grad()
def collect(model, tok, prompts, dev):
    """Last-token states at LAYERS; prompts are run in groups of equal token length (no padding)."""
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    out = np.zeros((len(LAYERS), len(prompts), model.config.hidden_size), np.float32)
    by_len = {}
    for i, x in enumerate(ids):
        by_len.setdefault(len(x), []).append(i)
    for L, idx in by_len.items():
        for s in range(0, len(idx), 64):
            b = idx[s:s + 64]
            hs = model(input_ids=torch.tensor([ids[i] for i in b], device=dev), output_hidden_states=True).hidden_states
            for j, l in enumerate(LAYERS):
                out[j, b] = hs[l][:, -1].float().cpu().numpy()
    return out


@torch.no_grad()
def score_tasks(model, tok, T, dev, bs=64):
    """Accuracy per domain: the candidate with the highest summed log-prob after the prompt."""
    seqs, meta = [], []
    for name, rows in T.items():
        for r, (prompt, cands, ans) in enumerate(rows):
            pid = tok(prompt, add_special_tokens=False)["input_ids"]
            for c, cand in enumerate(cands):
                cid = tok(" " + cand, add_special_tokens=False)["input_ids"]
                seqs.append((pid + cid, len(cid))); meta.append((name, r, c))
    score = np.zeros(len(seqs))
    by_len = {}
    for i, (ids, _) in enumerate(seqs):
        by_len.setdefault(len(ids), []).append(i)
    for L, idx in by_len.items():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            x = torch.tensor([seqs[i][0] for i in b], device=dev)
            m = max(seqs[i][1] for i in b)          # logits only where candidate tokens are predicted
            lp = torch.log_softmax(model(input_ids=x, logits_to_keep=m + 1).logits[:, :-1].float(), -1)
            tok_lp = lp.gather(-1, x[:, L - m:, None])[..., 0]
            for j, i in enumerate(b):
                score[i] = float(tok_lp[j, m - seqs[i][1]:].sum())
    best = {}
    for (name, r, c), sc in zip(meta, score):
        if (name, r) not in best or sc > best[(name, r)][1]:
            best[(name, r)] = (c, sc)
    return {name: dict(acc=float(np.mean([best[(name, r)][0] == rows[r][2] for r in range(len(rows))])),
                       chance=float(np.mean([1 / len(c) for _, c, _ in rows])), n=len(rows)) for name, rows in T.items()}


class LOO:
    """Leave-one-out R^2 for many targets on the same states (the eigendecomposition is shared); see loo_r2."""

    def __init__(self, H, alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0)):
        n = len(H)
        Hc = H - H.mean(0)
        ev, U = np.linalg.eigh(Hc @ Hc.T)
        sc = ev.sum() / n
        self.S = [(U * (ev / (ev + a * sc))) @ U.T + 1.0 / n for a in alphas]
        self.n = n

    def __call__(self, T):
        n = self.n
        sst = ((T - T.mean(0)) ** 2).sum(0) * (n / (n - 1)) ** 2
        return max(float(np.mean(1 - (((T - S @ T) / (1 - np.diag(S))[:, None]) ** 2).sum(0) / sst)) for S in self.S)


def loo_r2(H, T, alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Best-alpha leave-one-out R^2 (mean over columns) of ridge H -> T with an unpenalised intercept. The hat matrix
    is 11^T/n + Kc (Kc + a I)^-1 with Kc the centred kernel; the intercept term matters when n <= dim (without it the
    held-out item leaks through the centring and every target, permuted or not, scores R^2 = 1)."""
    n = len(H)
    Hc = H - H.mean(0)
    K = Hc @ Hc.T
    sc = np.trace(K) / n
    ev, U = np.linalg.eigh(K)
    Tc = T - T.mean(0)
    sst = (Tc ** 2).sum(0) * (n / (n - 1)) ** 2      # LOO-mean baseline: predicting each item by the others' mean
    best = -np.inf
    for a in alphas:
        S = (U * (ev / (ev + a * sc))) @ U.T + 1.0 / n
        resid = (T - S @ T) / (1 - np.diag(S))[:, None]
        best = max(best, float(np.mean(1 - (resid ** 2).sum(0) / sst)))
    return best


def cka(H, T):
    G, K = H @ H.T, T @ T.T
    return float((G * K).sum() / (np.linalg.norm(G) * np.linalg.norm(K) + 1e-12))


def basis(H, T):
    q, _ = np.linalg.qr(H.T @ T)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_null", type=int, default=N_NULL)
    ap.add_argument("--tiny", action="store_true", help="smoke test: a small random OLMo-2 with the given tokenizer, on CPU")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.tiny:
        from transformers import Olmo2Config, Olmo2ForCausalLM
        global LAYERS
        dev, LAYERS = "cpu", (0, 1, 2, 4)
        model = Olmo2ForCausalLM(Olmo2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                             num_attention_heads=4, num_key_value_heads=4)).eval()
    else:
        dev = "cuda"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    D = domains()
    names = list(D)
    rng = np.random.default_rng(0)
    res = {"tag": args.tag, "layers": LAYERS, "domains": {k: {"n": len(d["items"]), "kind": d["kind"], "primary": primary(d),
                                                             "targets": list(d["targets"])} for k, d in D.items()}}
    keep = {}
    for tmpl in ("neutral", "cued"):
        prompts, owner = [], []
        for k, d in D.items():
            for it in d["items"]:
                prompts.append(f"Q: {it}" if tmpl == "neutral" else d["cued"].format(it))
                owner.append(k)
        owner = np.array(owner)
        S = collect(model, tok, prompts, dev)
        S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)
        keep[tmpl] = S[[LAYERS.index(l) for l in (8, 16, 24) if l in LAYERS]].astype(np.float16)
        fits, overlap, overlap_null, removal = {}, [], [], []
        for li, L in enumerate(LAYERS):
            Hs = {k: S[li, owner == k] - S[li, owner == k].mean(0) for k in names}
            B, Bnull = {}, {}
            for k in names:
                d = D[k]
                loo = LOO(Hs[k])
                for tn, T in d["targets"].items():
                    r = loo(T)
                    perms = [rng.permutation(len(T)) for _ in range(args.n_null)]
                    nr = np.array([loo(T[p]) for p in perms])
                    c = cka(Hs[k], T); nc = np.array([cka(Hs[k], T[p]) for p in perms[:50]])
                    fits.setdefault(k, {}).setdefault(tn, []).append(dict(
                        r2=r, null=float(nr.mean()), null95=float(np.quantile(nr, .95)), p=float((nr >= r).mean()),
                        cka=c, cka_null=float(nc.mean())))
                T = d["targets"][primary(d)]
                B[k] = basis(Hs[k], T)
                Bnull[k] = [basis(Hs[k], T[rng.permutation(len(T))]) for _ in range(20)]
            ov = np.array([[float((B[a].T @ B[b]) .__pow__(2).sum() / min(B[a].shape[1], B[b].shape[1])) for b in names] for a in names])
            on = np.array([[float(np.mean([(x.T @ y).__pow__(2).sum() / min(x.shape[1], y.shape[1])
                                           for x, y in zip(Bnull[a], Bnull[b])])) for b in names] for a in names])
            rm = np.zeros((len(names), len(names)))
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    P = Hs[b] - (Hs[b] @ B[a]) @ B[a].T
                    rm[i, j] = loo_r2(P, D[b]["targets"][primary(D[b])])
            overlap.append(ov.tolist()); overlap_null.append(on.tolist()); removal.append(rm.tolist())
        res[tmpl] = dict(fits=fits, overlap=overlap, overlap_null=overlap_null, removal=removal, order=names)
        print(f"{args.tag} {tmpl} done ({time.time() - t0:.0f}s)", flush=True)
    res["tasks"] = score_tasks(model, tok, tasks(D), dev)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"atlas_{args.tag}.json").write_text(json.dumps(res))
    np.savez_compressed(out / f"states_{args.tag}.npz", neutral=keep["neutral"], cued=keep["cued"],
                        owner=np.array([k for k, d in D.items() for _ in d["items"]]), layers=np.array([8, 16, 24]))
    for k in names:
        f = res["cued"]["fits"][k][primary(D[k])]
        best = max(range(len(LAYERS)), key=lambda i: f[i]["r2"] - f[i]["null"])
        t = res["tasks"][k]
        print(f"  {k:13s} {primary(D[k]):10s} best L{LAYERS[best]:2d}  R2 {f[best]['r2']:.2f}  null {f[best]['null']:.2f}  "
              f"p {f[best]['p']:.3f} | task {t['acc']:.2f} (chance {t['chance']:.2f})")
    print(f"done {args.tag} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
