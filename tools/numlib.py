"""Shared pieces for the number-helix experiments (our own implementation, Python 3.9 compatible).

Design choices that differ from the collaborator's pipeline, on purpose:
- number representations are read at the operand position INSIDE the arithmetic prompt, not from a
  number presented alone at sequence position 0;
- helix maps are fitted directly in the residual stream (ridge, no StandardScaler/PCA) and scored by
  held-out R^2 against nulls;
- every layer, 31 included, is the output of the decoder layer (pre final norm), read and written with
  the same forward hooks.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = "Output ONLY a number."
PERIODS = (2, 5, 10, 100)


# ---------------------------------------------------------------- model, prompts, token positions

def load(path: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    return model, tok


def binary_prompt(a: int, b: int) -> str:
    return f"{PREFIX}{a}+{b}="


def three_prompt(a: int, b: int, c: int) -> str:
    return f"{PREFIX}{a}+{b}+{c}="


def positions(tok, kind: str) -> dict:
    """Token index of each slot; asserts that numbers are single tokens as expected."""
    if kind == "binary":
        toks = tok.convert_ids_to_tokens(tok(binary_prompt(23, 45))["input_ids"])
        assert toks[-4:] == ["23", "+", "45", "="], toks
        n = len(toks)
        return {"A": n - 4, "op": n - 3, "B": n - 2, "eq": n - 1}
    if kind == "three":
        toks = tok.convert_ids_to_tokens(tok(three_prompt(23, 45, 12))["input_ids"])
        assert toks[-6:] == ["23", "+", "45", "+", "12", "="], toks
        n = len(toks)
        return {"A": n - 6, "op1": n - 5, "B": n - 4, "op2": n - 3, "C": n - 2, "eq": n - 1}
    raise ValueError(kind)


def number_token_ids(tok, hi: int = 999) -> np.ndarray:
    ids = []
    for n in range(hi + 1):
        enc = tok(str(n), add_special_tokens=False)["input_ids"]
        assert len(enc) == 1, (n, enc)
        ids.append(enc[0])
    return np.array(ids)


def encode(tok, prompts: list, device) -> torch.Tensor:
    enc = tok(prompts, return_tensors="pt", padding=True)
    assert bool(enc["attention_mask"].all()), "prompts must share one token length"
    return enc["input_ids"].to(device)


# ---------------------------------------------------------------- hooks

def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _with_hidden(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextmanager
def capture(model, layers: list, pos: list, store: dict):
    """store[L] = residual after decoder layer L at token positions `pos`, float32 [B, len(pos), H]."""
    handles = [
        model.model.layers[L].register_forward_hook(
            lambda _m, _i, out, L=L: store.__setitem__(L, _hidden(out)[:, pos, :].detach().float())
        )
        for L in layers
    ]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch(model, layer: int, pos: int, delta: torch.Tensor | None = None, replace: torch.Tensor | None = None):
    """Edit the residual after decoder layer `layer` at token `pos`: replace it and/or add delta ([B, H])."""
    def hook(_m, _i, out):
        h = _hidden(out).clone()
        if replace is not None:
            h[:, pos, :] = replace.to(h.dtype)
        if delta is not None:
            h[:, pos, :] = h[:, pos, :] + delta.to(h.dtype)
        return _with_hidden(out, h)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def last_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(model(input_ids).logits[:, -1].float(), dim=-1)


@torch.no_grad()
def collect(model, input_ids: torch.Tensor, layers: list, pos: list, bs: int = 256) -> dict:
    """{L: np.float32 [N, len(pos), H]} for every layer in `layers`."""
    out = {L: [] for L in layers}
    for i in range(0, len(input_ids), bs):
        store = {}
        with capture(model, layers, pos, store):
            model(input_ids[i:i + bs])
        for L in layers:
            out[L].append(store[L].cpu())
    return {L: torch.cat(v).numpy() for L, v in out.items()}


# ---------------------------------------------------------------- helix features and maps

def features(x, periods=PERIODS, linear: bool = True):
    """Helix basis: cos/sin(2*pi*x/T) per period (sin dropped for T=2: it is 0 on integers) + x/100."""
    x = np.asarray(x, dtype=np.float64)
    cols, names = [], []
    for T in periods:
        cols.append(np.cos(2 * np.pi * x / T)); names.append(f"cos{T}")
        if T != 2:
            cols.append(np.sin(2 * np.pi * x / T)); names.append(f"sin{T}")
    if linear:
        cols.append(x / 100.0); names.append("lin")
    return np.stack(cols, axis=-1), names


def component_columns(names: list) -> dict:
    """Column groups for per-period ablations: T2, T5, T10, T100, lin."""
    groups = {}
    for i, n in enumerate(names):
        key = "lin" if n == "lin" else "T" + n[3:]
        groups.setdefault(key, []).append(i)
    return groups


def fit_map(F: np.ndarray, H: np.ndarray, lam: float = 1e-3):
    """Ridge fit H ~ mu + F @ W (F [n,k] features, H [n,d] states). Returns mu [d], W [k,d]."""
    F = F.astype(np.float64); H = H.astype(np.float64)
    fm, hm = F.mean(0), H.mean(0)
    Fc, Hc = F - fm, H - hm
    W = np.linalg.solve(Fc.T @ Fc + lam * np.eye(F.shape[1]), Fc.T @ Hc)
    return hm - fm @ W, W


def cv_r2(F: np.ndarray, H: np.ndarray, groups: np.ndarray | None = None, folds: int = 5, seed: int = 0) -> float:
    """Held-out R^2 of the ridge map, total variance over all dims; folds split by `groups` if given."""
    rng = np.random.default_rng(seed)
    keys = np.unique(groups) if groups is not None else np.arange(len(F))
    keys = rng.permutation(keys)
    sse = sst = 0.0
    for k in range(folds):
        test_keys = keys[k::folds]
        test = np.isin(groups, test_keys) if groups is not None else np.isin(np.arange(len(F)), test_keys)
        mu, W = fit_map(F[~test], H[~test])
        pred = mu + F[test] @ W
        sse += ((H[test] - pred) ** 2).sum()
        sst += ((H[test] - H[~test].mean(0)) ** 2).sum()
    return float(1.0 - sse / sst)


def row_space_projector(W: np.ndarray) -> np.ndarray:
    """Orthonormal basis Q [d, r] of the span of W's rows (the helix subspace in the residual stream)."""
    q, r = np.linalg.qr(W.T)
    keep = np.abs(np.diag(r)) > 1e-8 * np.abs(np.diag(r)).max()
    return q[:, keep]


# ---------------------------------------------------------------- probes (states -> features)

ALPHAS = tuple(10.0 ** k for k in range(-2, 7))


def probe_accuracy(X, y: np.ndarray, candidates: np.ndarray, folds: int = 5, seed: int = 0,
                   alphas=ALPHAS, device=None, groups: np.ndarray | None = None) -> dict:
    """Cross-validated exact decoding accuracy of integer target y from states X [n, d].

    Kernel ridge onto the helix features f(y); per training fold the ridge strength is picked by the
    closed-form leave-one-out error (one eigendecomposition per fold), the held-out fold is decoded as
    the nearest candidate in feature space. Runs on the GPU when there is one (float64).

    `groups`: rows sharing a group never straddle train and test. Pass the input that fully determines
    the state being probed (e.g. the (a, b) pair for positions before c): otherwise a probe can score by
    memorising repeated states instead of reading a computed quantity."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=device)
    Fy = torch.as_tensor(features(y)[0], dtype=torch.float64, device=device)
    Fc = torch.as_tensor(features(candidates)[0], dtype=torch.float64, device=device)
    cand = torch.as_tensor(candidates, device=device)
    yt = torch.as_tensor(y, device=device)
    rng = np.random.default_rng(seed)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    keys = rng.permutation(np.unique(g))
    correct = 0
    for k in range(folds):
        test_mask = np.isin(g, keys[k::folds])
        te = torch.as_tensor(np.flatnonzero(test_mask), device=device)
        tr = torch.as_tensor(np.flatnonzero(~test_mask), device=device)
        xm, ym = X[tr].mean(0), Fy[tr].mean(0)
        Xtr, Ytr = X[tr] - xm, Fy[tr] - ym
        K = Xtr @ Xtr.T
        lam, U = torch.linalg.eigh(K)
        lam = lam.clamp_min(0)
        UY = U.T @ Ytr
        best, best_err = None, None
        for a in alphas:
            h = lam / (lam + a)
            fitted = U @ (h[:, None] * UY)
            hii = (U ** 2) @ h
            err = (((Ytr - fitted) / (1 - hii).clamp_min(1e-6)[:, None]) ** 2).mean()
            if best_err is None or err < best_err:
                best, best_err = a, err
        A = U @ (UY / (lam + best)[:, None])
        pred = (X[te] - xm) @ (Xtr.T @ A) + ym
        dec = cand[torch.cdist(pred, Fc).argmin(1)]
        correct += int((dec == yt[te]).sum())
    return {"acc": correct / len(y), "n": int(len(y))}


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1))
