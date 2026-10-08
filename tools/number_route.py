"""P2: does the model do calendar / alphabet arithmetic through its number line? (the "number route")

"Arithmetic in the Wild" (arXiv 2605.01148) reports that Llama solves "Friday plus 3 days" by mapping the day to a
number, adding in base 10 and mapping back, rather than by rotating on a 7-circle. If OLMo-2 does the same, then
shaping the day/month manifold into a circle (concept_train.py) works against the circuit the model uses. Three
read-outs per checkpoint, all with permutation nulls:

1. sum route. At every 2nd layer, fit a ridge map from the last-token state of numeric prompts
   "Q: {a} plus {k} {unit} is" (a = 0..59) to the helix features of s = a + k and decode s by the nearest candidate
   in 0..199 (held-out accuracy: folds over a). Apply the same map to "Q: {item} plus {k} {unit} is":
     slope, r2       within each item, OLS of the decoded s on k (number route: slope ~ 1, r2 high)
     consistency     share of prompts whose (decoded s - k) equals the item's modal value (= the item's number)
     offsets         that modal value per item (Monday -> 1? January -> 1?); offset_slope = its slope on the index
     line_acc        decoded s == index + o + k for the best global o (consecutive numbering)
     mod_acc         decoded s == index + o + k (mod n) for the best o (cyclic domains: lands on the right item)
   Null: the same statistics with k permuted within each item (keeps the decoded values, breaks the link to k).
   The model's own answer accuracy on the domain prompts is reported, and the statistics are split by correctness.
2. operand route. The same map from "Q: {n}" (n = 0..199) to helix(n), applied to "Q: {item}": the item's number.
3. alignment. Centred cosine between item i and number n = i + o (o = 0, 1) in the input embedding, the output
   embedding, and the "Q: x" state of every probed layer, against 2000 item permutations; linear CKA between the
   item set and the number window as a rotation-invariant version. Number words one..twenty are the positive control.

Zero-shot, base checkpoints barely solve the domain prompts (<= 5%), so the sum route is also run with
--shots N (a prefix of N solved prompts of the same domain, never numeric ones) and, through analyze(),
on fine-tuned models (concept_train.py --route).

    python tools/number_route.py --out <dir> [--shots 4] <ckpt_dir> [<ckpt_dir> ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import numlib as nl
from concept_train import DAYS, LETTERS, MONTHS
from geom_diag import cka_np
from manifold_train import decoder_layers

WORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen twenty").split()
# items, unit word ("" = none), shifts k, cyclic?
DOMAINS = {
    "days": dict(items=DAYS, unit="days", ks=range(1, 41), cyclic=True),
    "months": dict(items=MONTHS, unit="months", ks=range(1, 41), cyclic=True),
    "letters": dict(items=LETTERS, unit="letters", ks=range(1, 13), cyclic=False),
    "numwords": dict(items=WORDS, unit="", ks=range(1, 21), cyclic=False),
}
ALPHAS = tuple(10.0 ** k for k in range(-2, 7))
CANDS = np.arange(200)
N_NULL = 200


def prompt(x, k, unit: str) -> str:
    return f"Q: {x} plus {k} {unit} is" if unit else f"Q: {x} plus {k} is"


@torch.no_grad()
def run(model, tok, texts, layers, dev, bs=256):
    """({L: [n, d]} last-token states after decoder layer L, [n] greedy next token). Texts share one token length."""
    dls = decoder_layers(model)
    got = {}
    hooks = [dls[L].register_forward_hook(lambda _m, _i, o, L=L: got.__setitem__(L, nl._hidden(o)[:, -1].float().cpu()))
             for L in layers]
    out = {L: [] for L in layers}
    nxt = []
    try:
        for i in range(0, len(texts), bs):
            ids = nl.encode(tok, texts[i:i + bs], dev)
            nxt.append(model(input_ids=ids, logits_to_keep=1).logits[:, -1].argmax(-1).cpu())
            for L in layers:
                out[L].append(got[L])
    finally:
        for h in hooks:
            h.remove()
    return {L: torch.cat(v).numpy() for L, v in out.items()}, torch.cat(nxt).numpy()


def ridge(X: np.ndarray, Y: np.ndarray, dev):
    """Dual ridge X -> Y with the strength chosen by leave-one-out; returns predict(Xnew)."""
    Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
    Yt = torch.as_tensor(Y, dtype=torch.float64, device=dev)
    xm, ym = Xt.mean(0), Yt.mean(0)
    Xc, Yc = Xt - xm, Yt - ym
    lam, U = torch.linalg.eigh(Xc @ Xc.T)
    lam = lam.clamp_min(0)
    UY = U.T @ Yc
    best, best_err = None, None
    for a in ALPHAS:
        h = lam / (lam + a)
        err = (((Yc - U @ (h[:, None] * UY)) / (1 - (U ** 2) @ h).clamp_min(1e-6)[:, None]) ** 2).mean()
        if best_err is None or err < best_err:
            best, best_err = a, err
    W = Xc.T @ (U @ (UY / (lam + best)[:, None]))

    def predict(Xn):
        Xn = torch.as_tensor(Xn, dtype=torch.float64, device=dev)
        return ((Xn - xm) @ W + ym).cpu().numpy()
    return predict


def decode(P: np.ndarray, fstd: np.ndarray) -> np.ndarray:
    """Nearest candidate 0..199 in helix-feature space (features scaled by their training std)."""
    C = nl.features(CANDS)[0] / fstd
    d = ((P / fstd)[:, None, :] - C[None]) ** 2
    return CANDS[d.sum(-1).argmin(1)]


def fit_decoder(X, s, groups, dev, folds=5):
    """Fit on all rows; held-out exact / within-1 accuracy with folds over `groups`."""
    F = nl.features(s)[0]
    fstd = F.std(0) + 1e-9
    ug = np.unique(groups)
    fold_of = {g: i % folds for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
    fo = np.array([fold_of[g] for g in groups])
    pred = np.zeros(len(s), dtype=np.int64)
    for f in range(folds):
        te = fo == f
        pred[te] = decode(ridge(X[~te], F[~te], dev)(X[te]), fstd)
    full = ridge(X, F, dev)
    return (lambda Xn: decode(full(Xn), fstd)), float((pred == s).mean()), float((np.abs(pred - s) <= 1).mean())


def route_stats(sh: np.ndarray, idx: np.ndarray, k: np.ndarray, n: int, cyclic: bool) -> dict:
    """Number-route statistics of decoded sums `sh` for prompts (item index idx, shift k)."""
    slopes, r2s, cons, offs = [], [], [], []
    for i in range(n):
        m = idx == i
        if m.sum() < 3:
            continue
        x, y = k[m].astype(float), sh[m].astype(float)
        xc = x - x.mean()
        b = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
        res = y - y.mean() - b * xc
        tot = ((y - y.mean()) ** 2).sum()
        slopes.append(b); r2s.append(1 - (res ** 2).sum() / tot if tot > 0 else 0.0)
        d = sh[m] - k[m]
        vals, cnt = np.unique(d, return_counts=True)
        cons.append(cnt.max() / m.sum()); offs.append(int(vals[cnt.argmax()]))
    out = {"slope": float(np.mean(slopes)), "r2": float(np.mean(r2s)), "consistency": float(np.mean(cons)), "offsets": offs}
    ii = np.arange(len(offs))
    out["offset_slope"] = float(np.polyfit(ii, offs, 1)[0]) if len(offs) > 2 else float("nan")
    os_ = range(0, n + 3)
    out["line_acc"] = float(max((sh == idx + o + k).mean() for o in os_))
    if cyclic:
        out["mod_acc"] = float(max(((sh - idx - o - k) % n == 0).mean() for o in range(n)))
    return out


def null_stats(sh, idx, k, n, cyclic, rng) -> dict:
    keys = ("slope", "r2", "consistency", "line_acc") + (("mod_acc",) if cyclic else ())
    acc = {key: [] for key in keys}
    for _ in range(N_NULL):
        kp = k.copy()
        for i in range(n):
            m = np.flatnonzero(idx == i)
            kp[m] = k[rng.permutation(m)]
        r = route_stats(sh, idx, kp, n, cyclic)
        for key in keys:
            acc[key].append(r[key])
    return {key: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))} for key, v in acc.items()}


def centred_cos(A: np.ndarray, B: np.ndarray) -> float:
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return float((A * B).sum(1).mean())


def alignment(E_items: np.ndarray, E_nums: np.ndarray, rng) -> dict:
    """Item i vs number i + o: centred cosine and CKA, each against item permutations."""
    n = len(E_items)
    out = {}
    perms = [rng.permutation(n) for _ in range(2000)]
    for o in (0, 1):
        B = E_nums[o:o + n]
        obs = centred_cos(E_items, B)
        null = np.array([centred_cos(E_items[p], B) for p in perms])
        ck = cka_np(E_items, B)
        cnull = np.array([cka_np(E_items[p], B) for p in perms[:300]])
        out[f"o{o}"] = {"cos": obs, "cos_null_mean": float(null.mean()), "cos_p": float(((null >= obs).sum() + 1) / (len(null) + 1)),
                        "cka": ck, "cka_null_mean": float(cnull.mean()), "cka_p": float(((cnull >= ck).sum() + 1) / (len(cnull) + 1))}
    return out


def shot_prefix(dname: str, shots: int) -> str:
    """Few-shot prefix of `shots` solved prompts of the domain itself (never numeric ones, which would prime the
    number route); fixed per domain, shared by the numeric fit prompts and the domain prompts."""
    if shots <= 0:
        return ""
    D = DOMAINS[dname]
    items, n = D["items"], len(D["items"])
    rng = np.random.default_rng(11)
    lines = []
    while len(lines) < shots:
        i, k = int(rng.integers(n)), int(rng.choice(list(D["ks"])))
        if dname == "numwords":
            ans = str(i + 1 + k)
        else:
            j = (i + k) % n if D["cyclic"] else i + k
            if j >= n:
                continue
            ans = items[j]
        lines.append(f"{prompt(items[i], k, D['unit'])} {ans}\n")
    return "".join(lines)


def analyze(model, tok, doms, layers, dev, rng, shots: int = 0, log=print) -> dict:
    """All three read-outs for one model (a base checkpoint, or a fine-tuned one with its adapter active)."""
    t0 = time.time()
    num_ids = nl.number_token_ids(tok, 199)
    E_in = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    E_out = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    res = {"layers": layers, "shots": shots, "operand": {}, "sum": {}, "align": {}}

    # operand route: "Q: n" -> helix(n), applied to "Q: item"
    S_num, _ = run(model, tok, [f"Q: {v}" for v in CANDS], layers, dev)
    op_dec = {L: fit_decoder(S_num[L], CANDS, CANDS, dev, folds=10) for L in layers}
    for dname in doms:
        D = DOMAINS[dname]
        items, n = D["items"], len(D["items"])
        ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in items]
        assert all(len(t) == 1 for t in ids), (dname, ids)
        ids = np.array([t[0] for t in ids])
        S_it, _ = run(model, tok, [f"Q: {w}" for w in items], layers, dev)
        res["operand"][dname] = {}
        for L in layers:
            dec, acc, acc1 = op_dec[L]
            v = dec(S_it[L])
            rho = float(np.corrcoef(np.argsort(np.argsort(v)), np.arange(n))[0, 1]) if np.ptp(v) > 0 else 0.0
            res["operand"][dname][f"L{L}"] = {"probe_acc": acc, "probe_acc1": acc1, "decoded": v.tolist(), "spearman": rho}

        # alignment: embeddings and "Q: x" states, item i vs number i + o
        al = {"emb_in": alignment(E_in[ids], E_in[num_ids[:n + 1]], rng),
              "emb_out": alignment(E_out[ids], E_out[num_ids[:n + 1]], rng)}
        for L in layers:
            al[f"L{L}"] = alignment(S_it[L], S_num[L][:n + 1], rng)
        res["align"][dname] = al

        # sum route (the few-shot prefix, if any, precedes both the numeric fit prompts and the domain prompts)
        pre = shot_prefix(dname, shots)
        ks = np.array(list(D["ks"]))
        A, K = np.meshgrid(np.arange(60), ks, indexing="ij")
        A, K = A.ravel(), K.ravel()
        S_fit, _ = run(model, tok, [pre + prompt(a, k, D["unit"]) for a, k in zip(A, K)], layers, dev)
        I, KK = np.meshgrid(np.arange(n), ks, indexing="ij")
        I, KK = I.ravel(), KK.ravel()
        S_dom, nxt = run(model, tok, [pre + prompt(items[i], k, D["unit"]) for i, k in zip(I, KK)], layers, dev)
        if dname == "numwords":
            correct, ok = None, None   # the answer starts with a bare space token
        else:
            ans = (I + KK) % n if D["cyclic"] else I + KK
            ok = ans < n
            correct = ok & (nxt == ids[np.minimum(ans, n - 1)])
        rs = {"model_acc": None if correct is None else float(correct[ok].mean()), "prefix": pre, "layers": {}}
        for L in layers:
            dec, acc, acc1 = fit_decoder(S_fit[L], A + K, A, dev)
            sh = dec(S_dom[L])
            st = route_stats(sh, I, KK, n, D["cyclic"])
            st.update(probe_acc=acc, probe_acc1=acc1, null=null_stats(sh, I, KK, n, D["cyclic"], rng),
                      decoded_hist=np.bincount(sh, minlength=200).tolist())
            if correct is not None and correct.sum() >= 20 and (ok & ~correct).sum() >= 20:
                st["by_correct"] = {c: route_stats(sh[m], I[m], KK[m], n, D["cyclic"])
                                    for c, m in (("correct", correct), ("wrong", ok & ~correct))}
            rs["layers"][f"L{L}"] = st
        res["sum"][dname] = rs
        log(f"route {dname}: model_acc {rs['model_acc']} ({time.time() - t0:.0f}s)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--shots", type=int, default=0, help="few-shot prefix of solved domain prompts for the sum route")
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    doms = args.domains.split(",")
    for ck in args.ckpts:
        name = Path(ck).name
        path = out / f"route_{name}.json"
        if path.exists():
            continue
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16).to(dev)
        model.config.use_cache = False
        nL = model.config.num_hidden_layers
        res = {"ckpt": ck, "n_layers": nL}
        res.update(analyze(model, tok, doms, list(range(2, nL, 2)), dev, np.random.default_rng(0), args.shots,
                           log=lambda s: print(f"{name} {s}", flush=True)))
        path.write_text(json.dumps(res, indent=1))
        del model
        gc.collect()
        torch.cuda.empty_cache() if dev == "cuda" else None
    summarize(out)


def summarize(out: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("route_*.json"))]
    lines = [f"# P2: number route (shots {rows[0].get('shots', 0) if rows else 0})\n\n## sum route (layer with the highest consistency; null = k permuted within item)\n\n",
             "| ckpt | domain | model acc | layer | probe acc (numeric) | slope | r2 (null p95) | consistency (null p95) | "
             "line acc (null p95) | mod acc (null p95) | offsets |\n|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        for d, rs in r["sum"].items():
            Lk, st = max(rs["layers"].items(), key=lambda kv: kv[1]["consistency"])
            nu = st["null"]
            mod = f"{st['mod_acc']:.2f} ({nu['mod_acc']['p95']:.2f})" if "mod_acc" in st else "-"
            macc = "-" if rs["model_acc"] is None else f"{rs['model_acc']:.2f}"
            lines.append(f"| {Path(r['ckpt']).name} | {d} | {macc} | {Lk} | {st['probe_acc']:.2f} | {st['slope']:.2f} | "
                         f"{st['r2']:.2f} ({nu['r2']['p95']:.2f}) | {st['consistency']:.2f} ({nu['consistency']['p95']:.2f}) | "
                         f"{st['line_acc']:.2f} ({nu['line_acc']['p95']:.2f}) | {mod} | {st['offsets'][:12]} |\n")
    lines.append("\n## operand route (\"Q: item\" decoded on the number line; layer with the highest |spearman|)\n\n"
                 "| ckpt | domain | layer | probe acc (numeric) | spearman | decoded |\n|---|---|---|---|---|---|\n")
    for r in rows:
        for d, od in r["operand"].items():
            Lk, st = max(od.items(), key=lambda kv: abs(kv[1]["spearman"]))
            lines.append(f"| {Path(r['ckpt']).name} | {d} | {Lk} | {st['probe_acc']:.2f} | {st['spearman']:.2f} | {st['decoded'][:12]} |\n")
    lines.append("\n## alignment: item i vs number i + o (centred cosine, p vs 2000 permutations; CKA, p vs 300)\n\n"
                 "| ckpt | domain | site | o | cos (null) p | CKA (null) p |\n|---|---|---|---|---|---|\n")
    for r in rows:
        for d, al in r["align"].items():
            sites = ["emb_in", "emb_out"] + [max((s for s in al if s.startswith("L")),
                                                 key=lambda s: max(al[s]["o0"]["cos"], al[s]["o1"]["cos"]))]
            for s in sites:
                for o in ("o0", "o1"):
                    a = al[s][o]
                    lines.append(f"| {Path(r['ckpt']).name} | {d} | {s} | {o[1]} | {a['cos']:.3f} ({a['cos_null_mean']:.3f}) {a['cos_p']:.3f} | "
                                 f"{a['cka']:.2f} ({a['cka_null_mean']:.2f}) {a['cka_p']:.3f} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
