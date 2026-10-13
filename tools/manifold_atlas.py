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
  controls token_sum = the target decoded from the summed input embeddings of the item's tokens; token_sum_random
           = the same with a fixed random vector per token id (what the tokenisation alone gives, e.g. "194"+"5")
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

# primary target per kind. For Z_n NOT all frequencies: with every k the centred target spans all n - 1 directions,
# i.e. it is the one-hot code, so a held-out item can never be predicted whatever the geometry (R^2 = 0 by
# construction). The circle (k = 1) is the geometric claim (Engels et al.); pitch / keys use the chromatic circle,
# colours the hue circle.
# atlas v2 (audit 2026-10-13): ordered domains use the LINE (v1 used the cubic curve, whose degree-2/3 columns are
# not decodable from a perfectly linear code: ceiling 0.08-0.31 depending on n) and places the unit SPHERE (v1:
# degree-2 harmonics, ceiling ~0.4-0.6 for a perfect 3-D code); people use the rank of the birth year (the year axis
# is dominated by the 13 people born before 1000). Tree targets are read with the pooled R^2 (r2p).
PRIMARY = {"helix": "helix", "cyclic": "circle", "linear": "line", "sphere": "sphere", "plane": "plane",
           "magnitude": "log", "tree": "tree", "kin": "generation", "scalar": "within"}
PRIMARY_OVERRIDE = {"pitch": "chromatic", "keys": "chromatic", "colours": "hue", "people": "rank"}
POOLED_KINDS = ("tree",)   # kinds whose headline score is the pooled R^2 (column weights matter)
LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32)   # OLMo-2-7B; other depths: 9 evenly spaced layers (set in main)
N_NULL = 200


def primary(d, name=None):
    if name in PRIMARY_OVERRIDE:
        return PRIMARY_OVERRIDE[name]
    return PRIMARY[d["kind"]] if PRIMARY[d["kind"]] in d["targets"] else next(iter(d["targets"]))


def token_sum(model, tok, items, random=False, dim=1024):
    """Controls without context or layers: the sum of the item's token embeddings. random=True uses a fixed random
    vector per token id instead, so a target decodable from it is decodable from the TOKENISATION alone (e.g. "1945" =
    "194" + "5", "C#" = "C" + "#"), not evidence of a learned manifold. random=False uses the model's own input
    embeddings (single-token items: geometry already learned in the embedding matrix counts)."""
    ids = [tok(" " + it, add_special_tokens=False)["input_ids"] for it in items]
    if random:
        return np.stack([sum(np.random.default_rng(t).normal(size=dim) for t in x) for x in ids])
    E = model.get_input_embeddings().weight
    return np.stack([E[x].float().sum(0).detach().cpu().numpy() for x in ids])


@torch.no_grad()
def collect(model, tok, prompts, dev):
    """Last-token states at LAYERS; prompts are run in groups of equal token length (no padding)."""
    ids = [tok(p)["input_ids"] for p in prompts]   # default special tokens: BOS for Mistral, nothing for the others
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
def _cand_logprobs(model, tok, pairs, dev, bs=64):
    """Summed log-prob of each continuation after its prompt: pairs = [(prompt, continuation)]. The prompt is
    tokenised with the default special tokens (BOS where the model was trained with one), the continuation without."""
    seqs = []
    for prompt, cont in pairs:
        pid = tok(prompt)["input_ids"]
        cid = tok(cont, add_special_tokens=False)["input_ids"]
        seqs.append((pid + cid, len(cid)))
    score = np.zeros(len(seqs))
    by_len = {}
    for i, (ids, _) in enumerate(seqs):
        by_len.setdefault(len(ids), []).append(i)
    for L, idx in by_len.items():
        for s0 in range(0, len(idx), bs):
            b = idx[s0:s0 + bs]
            x = torch.tensor([seqs[i][0] for i in b], device=dev)
            m = max(seqs[i][1] for i in b)          # logits only where continuation tokens are predicted
            lp = torch.log_softmax(model(input_ids=x, logits_to_keep=m + 1).logits[:, :-1].float(), -1)
            tok_lp = lp.gather(-1, x[:, L - m:, None])[..., 0]
            for j, i in enumerate(b):
                score[i] = float(tok_lp[j, m - seqs[i][1]:].sum())
    return score, np.array([n for _, n in seqs])


@torch.no_grad()
def score_tasks(model, tok, T, dev, bs=64):
    """Per domain: accuracy of the highest summed log-prob candidate (acc), of the length-normalised log-prob
    (acc_norm), of the calibrated score log p(c | question) - log p(c | content-free question) (acc_cal, Zhao et al.
    2021), and of the content-free score alone (acc_prior: what the candidate prior gets without the question; far
    from chance = the test can be passed without the domain's knowledge). Candidates are scored as " c\n" (the
    shots' answer format; also stops a candidate that is a token prefix of another from winning by default).
    rows = per test row [answer, [log p], [log p content-free], [n tokens]] so that any calibration can be redone."""
    pairs, pairs_cf, meta = [], [], []
    for name, rows in T.items():
        for r, (prompt, cands, ans, cf) in enumerate(rows):
            for c, cand in enumerate(cands):
                pairs.append((prompt, " " + cand + "\n")); pairs_cf.append((cf, " " + cand + "\n")); meta.append((name, r, c))
    lp, ntok = _cand_logprobs(model, tok, pairs, dev, bs)
    lp_cf, _ = _cand_logprobs(model, tok, pairs_cf, dev, bs)
    per = {}
    for (name, r, c), a, b, n in zip(meta, lp, lp_cf, ntok):
        per.setdefault(name, {}).setdefault(r, []).append((a, b, int(n)))
    out = {}
    for name, rows in T.items():
        recs, hit = [], {"acc": [], "acc_norm": [], "acc_cal": [], "acc_prior": []}
        for r, (_, cands, ans, _) in enumerate(rows):
            sc = np.array(per[name][r])
            hit["acc"].append(int(np.argmax(sc[:, 0])) == ans)
            hit["acc_norm"].append(int(np.argmax(sc[:, 0] / sc[:, 2])) == ans)
            hit["acc_cal"].append(int(np.argmax(sc[:, 0] - sc[:, 1])) == ans)
            hit["acc_prior"].append(int(np.argmax(sc[:, 1])) == ans)
            recs.append([ans, sc[:, 0].round(4).tolist(), sc[:, 1].round(4).tolist(), sc[:, 2].astype(int).tolist()])
        out[name] = dict({k: float(np.mean(v)) for k, v in hit.items()},
                         chance=float(np.mean([1 / len(c) for _, c, _, _ in rows])), n=len(rows), rows=recs)
    return out


class LOO:
    """Leave-one-out R^2 for many targets on the same states (the eigendecomposition is shared); see loo_r2."""

    def __init__(self, H, alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0)):
        n = len(H)
        Hc = H - H.mean(0)
        ev, U = np.linalg.eigh(Hc @ Hc.T)
        ev = np.clip(ev, 0, None)
        sc = max(ev.sum() / n, 1e-12)   # identical states (e.g. layer 0 of "13:00" sees only "00"): S = 11^T / n
        self.S = [(U * (ev / (ev + a * sc))) @ U.T + 1.0 / n for a in alphas]
        self.n = n

    def __call__(self, T):
        return self.both(T)[0]

    def both(self, T):
        """(mean over columns, pooled 1 - sum SSE / sum SST), each at its own best alpha. The pooled version weights
        columns by their variance (so the weights of a tree code matter; a column with two items does not count as
        much as a class split)."""
        n = self.n
        sst = ((T - T.mean(0)) ** 2).sum(0) * (n / (n - 1)) ** 2
        rm, rp = -np.inf, -np.inf
        for S in self.S:
            sse = (((T - S @ T) / (1 - np.diag(S))[:, None]) ** 2).sum(0)
            rm = max(rm, float(np.mean(1 - sse / sst)))
            rp = max(rp, float(1 - sse.sum() / sst.sum()))
        return rm, rp


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
    """Orthonormal basis of the directions that carry T (PLS: span of H^T T), rank-revealing (a rank-deficient T, e.g.
    a centred one-hot code, must not add numerical-noise directions)."""
    u, sv, _ = np.linalg.svd(H.T @ T, full_matrices=False)
    return u[:, sv > 1e-6 * max(sv.max(), 1e-12)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_null", type=int, default=N_NULL)
    ap.add_argument("--tiny", action="store_true", help="smoke test: a small random OLMo-2 with the given tokenizer, on CPU")
    ap.add_argument("--random_init", action="store_true", help="negative control: the --model architecture with random weights")
    ap.add_argument("--only", default=None, help="smoke tests: comma-separated subset of domains")
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
    elif args.random_init:
        from transformers import AutoConfig
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model), torch_dtype=torch.bfloat16).to(dev).eval()
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    if not args.tiny:   # 9 evenly spaced layers 0..L (= 0, 4, .., 32 for the 32-layer OLMo-2-7B)
        nl_ = model.config.num_hidden_layers
        LAYERS = tuple(sorted({int(round(i * nl_ / 8)) for i in range(9)}))
    keep_layers = [LAYERS[len(LAYERS) * q // 4] for q in (1, 2, 3)]   # states saved at 1/4, 1/2, 3/4 depth
    print(f"model {args.model} ({type(model).__name__}, {model.config.num_hidden_layers} layers); layers {LAYERS}; "
          f"states saved at {keep_layers}", flush=True)
    D = domains()
    if args.only:
        D = {k: D[k] for k in args.only.split(",")}
    names = list(D)
    rng = np.random.default_rng(0)
    res = {"tag": args.tag, "layers": LAYERS, "args": vars(args), "model_class": type(model).__name__,
           "bos_added": tok("Q: x")["input_ids"] != tok("Q: x", add_special_tokens=False)["input_ids"],
           "domains": {k: {"n": len(d["items"]), "kind": d["kind"], "primary": primary(d, k),
                           "pooled": d["kind"] in POOLED_KINDS, "targets": list(d["targets"])} for k, d in D.items()}}
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
        keep[tmpl] = S[[LAYERS.index(l) for l in keep_layers]].astype(np.float16)
        fits, overlap, overlap_null, removal, removal_null = {}, [], [], [], []
        perms = {(k, tn): [rng.permutation(len(T)) for _ in range(args.n_null)] for k, d in D.items() for tn, T in d["targets"].items()}
        nulls = {key: [] for key in perms}
        nulls_p = {key: [] for key in perms}
        for li, L in enumerate(LAYERS):
            Hs = {k: S[li, owner == k] - S[li, owner == k].mean(0) for k in names}
            B, Bnull = {}, {}
            for k in names:
                d = D[k]
                loo = LOO(Hs[k])
                for tn, T in d["targets"].items():
                    r, rp = loo.both(T)
                    nb = np.array([loo.both(T[p]) for p in perms[(k, tn)]])
                    nr, nrp = nb[:, 0], nb[:, 1]
                    nulls[(k, tn)].append(nr); nulls_p[(k, tn)].append(nrp)
                    c = cka(Hs[k], T); nc = np.array([cka(Hs[k], T[p]) for p in perms[(k, tn)][:50]])
                    N = len(nr)
                    fits.setdefault(k, {}).setdefault(tn, []).append(dict(
                        r2=r, null=float(nr.mean()), null95=float(np.quantile(nr, .95)), p=float(((nr >= r).sum() + 1) / (N + 1)),
                        r2p=rp, null_p=float(nrp.mean()), null95_p=float(np.quantile(nrp, .95)),
                        p_p=float(((nrp >= rp).sum() + 1) / (N + 1)), cka=c, cka_null=float(nc.mean())))
                T = d["targets"][primary(d, k)]
                B[k] = basis(Hs[k], T)
                Bnull[k] = [basis(Hs[k], T[rng.permutation(len(T))]) for _ in range(20)]
            ov = np.array([[float((B[a].T @ B[b]) .__pow__(2).sum() / min(B[a].shape[1], B[b].shape[1])) for b in names] for a in names])
            on = np.array([[float(np.mean([(x.T @ y).__pow__(2).sum() / min(x.shape[1], y.shape[1])
                                           for x, y in zip(Bnull[a], Bnull[b])])) for b in names] for a in names])
            rm, rm_null = np.zeros((len(names), len(names))), np.zeros((len(names), len(names)))
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    Tb = D[b]["targets"][primary(D[b], b)]
                    P = Hs[b] - (Hs[b] @ B[a]) @ B[a].T
                    rm[i, j] = loo_r2(P, Tb)
                    # null: remove a basis of the same kind fitted to a's PERMUTED target (same dimension, same
                    # anisotropy of the state space) -- what removal costs b without a real shared subspace
                    rm_null[i, j] = np.mean([loo_r2(Hs[b] - (Hs[b] @ x) @ x.T, Tb) for x in Bnull[a][:3]])
            overlap.append(ov.tolist()); overlap_null.append(on.tolist()); removal.append(rm.tolist())
            removal_null.append(rm_null.tolist())
        best_layer = {}   # best layer vs the best-layer null (max over layers per permutation): no selection bonus
        for (k, tn), arr in nulls.items():
            entry = {}
            for key, nl_arr, sfx in (("r2", arr, ""), ("r2p", nulls_p[(k, tn)], "_p")):
                obs = np.array([f[key] for f in fits[k][tn]])
                nmax = np.max(np.stack(nl_arr), 0)
                entry.update({key: float(obs.max()), "layer" + sfx: LAYERS[int(obs.argmax())], "null" + sfx: float(nmax.mean()),
                              "null95" + sfx: float(np.quantile(nmax, .95)),
                              "p" + sfx: float(((nmax >= obs.max()).sum() + 1) / (len(nmax) + 1))})
            best_layer.setdefault(k, {})[tn] = entry
        res[tmpl] = dict(fits=fits, best=best_layer, overlap=overlap, overlap_null=overlap_null, removal=removal,
                         removal_null=removal_null, order=names)
        print(f"{args.tag} {tmpl} done ({time.time() - t0:.0f}s)", flush=True)
    res["tasks"] = score_tasks(model, tok, tasks(D), dev)
    for key, rnd in (("token_sum", False), ("token_sum_random", True)):   # controls: embeddings only / tokenisation only
        res[key] = {}
        for k, d in D.items():
            E = token_sum(model, tok, d["items"], random=rnd)
            E = (E - E.mean(0)) / (E.std(0) + 1e-6)
            loo = LOO(E - E.mean(0))
            res[key][k] = {tn: dict(r2=loo(T), null=float(np.mean([loo(T[rng.permutation(len(T))]) for _ in range(50)])))
                           for tn, T in d["targets"].items()}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"atlas_{args.tag}.json").write_text(json.dumps(res))
    np.savez_compressed(out / f"states_{args.tag}.npz", neutral=keep["neutral"], cued=keep["cued"],
                        owner=np.array([k for k, d in D.items() for _ in d["items"]]), layers=np.array(keep_layers))
    for k in names:
        b = res["cued"]["best"][k][primary(D[k], k)]
        t = res["tasks"][k]
        ts = res["token_sum"][k][primary(D[k], k)]; tr = res["token_sum_random"][k][primary(D[k], k)]
        print(f"  {k:14s} {primary(D[k], k):10s} best L{b['layer']:2d}  R2 {b['r2']:.2f} (pooled {b['r2p']:.2f})  best-layer null "
              f"{b['null']:.2f} (p95 {b['null95']:.2f}) p {b['p']:.3f} | emb-sum {ts['r2']:.2f} rand-tok-sum {tr['r2']:.2f} | "
              f"task n={t['n']} acc {t['acc']:.2f} cal {t['acc_cal']:.2f} prior-only {t['acc_prior']:.2f} (chance {t['chance']:.2f})")
    print(f"done {args.tag} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
