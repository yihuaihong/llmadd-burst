"""P1: spectral vs geometric structure of number representations across checkpoints.

Fu et al. (COLM 2026, arXiv 2604.20817) show that Fourier spikes at T = 2, 5, 10 appear in almost any number
embedding (even in raw token frequencies), and that a spike is necessary but not sufficient for the numbers to
be linearly separable by n mod T. Linear CKA to the helix may mostly track the spectral part. This script
measures both, plus CKA, on the same representations, for every checkpoint passed:

  spectral  Phi_T  = share of the non-DC Fourier power (along n) at frequency 1/T, and its ratio to the median
                     frequency ("spike");
  geometric kappa_T = Cohen's kappa of a cross-validated ridge classifier predicting n mod T (T = 2, 5, 10, and
                     3, 7 as non-decimal controls) and the tens digit;
  shape     CKA to the helix and to the digit code on 10..99 (the target of the training loss).

Representations: the input-embedding rows of the number tokens, and the operand state of "Q: x" (the state the
training loss shapes) at layers 1/8, 1/4, 3/8, 1/2, 3/4 of the depth. Number sets: digits 0..999, and digits
1..20 next to the number words one..twenty (same values, different surface form).

    python tools/geom_diag.py --out <dir> <ckpt_dir> [<ckpt_dir> ...]
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
from manifold_train import cka, decoder_layers, geom_features

WORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen twenty").split()
ALPHAS = tuple(10.0 ** k for k in range(-2, 7))


def fourier(X: np.ndarray, periods) -> dict:
    """X rows are n = 0..N-1 (consecutive). Share of the non-DC power at 1/T and the spike ratio to the median."""
    X = X - X.mean(0, keepdims=True)
    N = len(X)
    P = (np.abs(np.fft.rfft(X, axis=0)) ** 2).sum(1)[1:]       # k = 1..N/2
    total, med = P.sum(), np.median(P)
    out = {}
    for T in periods:
        if N % T == 0 and N // T <= len(P):
            p = P[N // T - 1]
            out[f"phi_{T}"] = float(p / total)
            out[f"spike_{T}"] = float(p / med)
    return out


def kappa(X: np.ndarray, y: np.ndarray, folds: int = 10, seed: int = 0, dev="cpu") -> float:
    """Cohen's kappa of a cross-validated ridge classifier (one-hot targets, ridge strength by leave-one-out)."""
    Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
    classes = np.unique(y)
    Y = torch.as_tensor((y[:, None] == classes[None]).astype(np.float64), device=dev)
    keys = np.random.default_rng(seed).permutation(len(y))
    pred = np.zeros(len(y), dtype=np.int64)
    for k in range(folds):
        te_idx = keys[k::folds]
        mask = np.zeros(len(y), bool); mask[te_idx] = True
        tr = torch.as_tensor(np.flatnonzero(~mask), device=dev); te = torch.as_tensor(np.flatnonzero(mask), device=dev)
        xm, ym = Xt[tr].mean(0), Y[tr].mean(0)
        Xtr, Ytr = Xt[tr] - xm, Y[tr] - ym
        lam, U = torch.linalg.eigh(Xtr @ Xtr.T)
        lam = lam.clamp_min(0)
        UY = U.T @ Ytr
        best, best_err = None, None
        for a in ALPHAS:
            h = lam / (lam + a)
            err = (((Ytr - U @ (h[:, None] * UY)) / (1 - (U ** 2) @ h).clamp_min(1e-6)[:, None]) ** 2).mean()
            if best_err is None or err < best_err:
                best, best_err = a, err
        A = U @ (UY / (lam + best)[:, None])
        scores = (Xt[te] - xm) @ (Xtr.T @ A) + ym
        pred[mask] = classes[scores.argmax(1).cpu().numpy()]
    acc = (pred == y).mean()
    pe = sum((y == c).mean() * (pred == c).mean() for c in classes)
    return float((acc - pe) / (1 - pe + 1e-12))


def cka_np(X: np.ndarray, F: np.ndarray) -> float:
    F = F[:, F.std(0) > 1e-9]
    F = (F - F.mean(0)) / F.std(0)
    Ft = torch.tensor(F, dtype=torch.float32)
    return float(cka(torch.tensor(X, dtype=torch.float32), Ft @ Ft.T))


def measure(X: np.ndarray, values: np.ndarray, dev) -> dict:
    """All metrics for representations X [n, d] of the numbers `values` (consecutive integers)."""
    out = fourier(X, (2, 5, 10, 100))
    for T in (2, 3, 5, 7, 10):
        out[f"kappa_mod{T}"] = kappa(X, values % T, dev=dev)
    if values.max() >= 20:
        out["kappa_tens"] = kappa(X, (values // 10) % 10, dev=dev)
    sel = (values >= 10) & (values <= 99)
    if sel.sum() == 90:
        perm = np.arange(100)
        out["cka_helix"] = cka_np(X[sel], geom_features("helix", values[sel], perm))
        out["cka_digit"] = cka_np(X[sel], geom_features("digit", values[sel], perm))
    return out


@torch.no_grad()
def states(model, tok, texts, layers, dev, bs=256) -> dict:
    """{L: [n, d]} state of the last token of each text after decoder layer L (texts must share one length)."""
    dls = decoder_layers(model)
    got = {}
    hooks = [dls[L].register_forward_hook(lambda _m, _i, o, L=L: got.__setitem__(L, nl._hidden(o)[:, -1].float().cpu()))
             for L in layers]
    out = {L: [] for L in layers}
    try:
        for i in range(0, len(texts), bs):
            ids = nl.encode(tok, texts[i:i + bs], dev)
            model(input_ids=ids, logits_to_keep=1)
            for L in layers:
                out[L].append(got[L])
    finally:
        for h in hooks:
            h.remove()
    return {L: torch.cat(v).numpy() for L, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ck in args.ckpts:
        name = Path(ck).name
        path = out / f"diag_{name}.json"
        if path.exists():
            continue
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16).to(dev)
        model.config.use_cache = False
        nL = model.config.num_hidden_layers
        layers = sorted({max(1, round(nL * f)) for f in (1 / 8, 1 / 4, 3 / 8, 1 / 2, 3 / 4)})
        num_ids = torch.tensor(nl.number_token_ids(tok, 999), device=dev)
        word_ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in WORDS]
        assert all(len(w) == 1 for w in word_ids), "number words must be single tokens"
        emb = model.get_input_embeddings().weight.detach()
        res = {"ckpt": ck, "n_layers": nL, "layers": layers, "digits": {}, "digits_1_20": {}, "words_1_20": {}}
        v1000 = np.arange(1000)
        res["digits"]["emb"] = measure(emb[num_ids].float().cpu().numpy(), v1000, dev)
        Sd = states(model, tok, [f"Q: {x}" for x in v1000], layers, dev)
        for L in layers:
            res["digits"][f"L{L}"] = measure(Sd[L], v1000, dev)
        # same values, two surface forms: "1".."20" vs "one".."twenty" (operand of "Q: x")
        v20 = np.arange(1, 21)
        res["digits_1_20"]["emb"] = measure(emb[num_ids[1:21]].float().cpu().numpy(), v20 - 1, dev)
        res["words_1_20"]["emb"] = measure(emb[torch.tensor([w[0] for w in word_ids], device=dev)].float().cpu().numpy(), v20 - 1, dev)
        Sw = states(model, tok, [f"Q: {w}" for w in WORDS], layers, dev)
        for L in layers:
            res["digits_1_20"][f"L{L}"] = measure(Sd[L][1:21], v20 - 1, dev)
            res["words_1_20"][f"L{L}"] = measure(Sw[L], v20 - 1, dev)
        path.write_text(json.dumps(res, indent=1))
        print(f"{name}: done ({time.time() - t0:.0f}s) emb kappa10 {res['digits']['emb']['kappa_mod10']:.2f} "
              f"phi10 {res['digits']['emb'].get('phi_10', float('nan')):.3f} cka {res['digits']['emb'].get('cka_helix', float('nan')):.2f}")
        del model, emb
        gc.collect()
        torch.cuda.empty_cache() if dev == "cuda" else None
    summarize(out)


def summarize(out: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("diag_*.json"))]
    lines = ["# P1: spectral vs geometric structure (digits 0..999; kappa = CV ridge classifier, Cohen's kappa)\n\n",
             "| ckpt | site | phi_10 | spike_10 | kappa mod2 / mod5 / mod10 | kappa mod3 / mod7 | kappa tens | CKA helix / digit |\n",
             "|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        for site, m in r["digits"].items():
            lines.append(f"| {Path(r['ckpt']).name} | {site} | {m.get('phi_10', float('nan')):.3f} | {m.get('spike_10', float('nan')):.1f} | "
                         f"{m['kappa_mod2']:.2f} / {m['kappa_mod5']:.2f} / {m['kappa_mod10']:.2f} | {m['kappa_mod3']:.2f} / {m['kappa_mod7']:.2f} | "
                         f"{m.get('kappa_tens', float('nan')):.2f} | {m.get('cka_helix', float('nan')):.2f} / {m.get('cka_digit', float('nan')):.2f} |\n")
    lines.append("\n## same values 1..20: digits vs number words (kappa mod2 / mod5, phi_10 / spike_10 over n = 1..20)\n\n")
    lines.append("| ckpt | site | digits kappa2 / kappa5 | words kappa2 / kappa5 | digits phi10 / spike10 | words phi10 / spike10 |\n|---|---|---|---|---|---|\n")
    for r in rows:
        for site in r["words_1_20"]:
            d, w = r["digits_1_20"][site], r["words_1_20"][site]
            lines.append(f"| {Path(r['ckpt']).name} | {site} | {d['kappa_mod2']:.2f} / {d['kappa_mod5']:.2f} | {w['kappa_mod2']:.2f} / {w['kappa_mod5']:.2f} | "
                         f"{d.get('phi_10', float('nan')):.3f} / {d.get('spike_10', float('nan')):.1f} | {w.get('phi_10', float('nan')):.3f} / {w.get('spike_10', float('nan')):.1f} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
