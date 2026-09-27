"""Number manifold in the EMBEDDING matrices across OLMo-2 checkpoints (CPU only, no forward passes).

For each revision only the safetensors shards holding model.embed_tokens.weight and lm_head.weight are
downloaded (then deleted). For the rows of the number tokens it reports, for the input embedding and the
output (unembedding, scaled by the final-norm weight) separately:
  - 10..99: linear CKA with the helix, digit (tens + units one-hot) and shuffled-helix coordinates, and
            number-held-out R^2 of both bases;
  - 100..999: CKA with a three-digit code (hundreds + tens + units one-hot) and its shuffled control;
  - neighbourhood: share of each number's 5 nearest neighbours (cosine) that are +-1 away / share its
            tens digit, for 10..99.

    python tools/emb_manifold.py --out <dir> [--revisions all|comma list]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_refs
from safetensors import safe_open

import numlib as nl

REPO = "allenai/OLMo-2-1124-7B"
STEPS = (150, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 150000, 250000, 400000, 500000, 700000, 850000, 928646)


def pick_revisions() -> list:
    refs = [b.name for b in list_repo_refs(REPO).branches]
    s1 = {int(re.search(r"step(\d+)", r).group(1)): r for r in refs if r.startswith("stage1-step")}
    out = [s1[min(s1, key=lambda s: abs(s - want))] for want in STEPS]
    s2 = sorted(r for r in refs if r.startswith("stage2-ingredient"))
    for i in (1, 2, 3):
        cand = [r for r in s2 if f"ingredient{i}-" in r]
        out.append(max(cand, key=lambda r: int(re.search(r"step(\d+)", r).group(1))))
    out.append("main")
    return list(dict.fromkeys(out))


def cka(H: np.ndarray, F: np.ndarray) -> float:
    H = H - H.mean(0)
    F = F[:, F.std(0) > 1e-9]; F = (F - F.mean(0)) / F.std(0)
    K, L = H @ H.T, F @ F.T
    return float((K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L)))


def metrics(E: np.ndarray) -> dict:
    """E: rows for numbers 0..999."""
    x = np.arange(10, 100); X = E[10:100]
    Fh = nl.features(x)[0]; Fd = np.concatenate([np.eye(10)[x // 10], np.eye(10)[x % 10]], 1)
    perm = np.random.default_rng(123).permutation(90)
    y = np.arange(100, 1000); Y = E[100:1000]
    F3 = np.concatenate([np.eye(10)[y // 100], np.eye(10)[(y // 10) % 10], np.eye(10)[y % 10]], 1)
    perm3 = np.random.default_rng(7).permutation(900)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :5]
    return {
        "cka_helix": cka(X, Fh), "cka_digit": cka(X, Fd), "cka_helix_shuf": cka(X, Fh[perm]),
        "r2_helix": nl.cv_r2(Fh, X), "r2_digit": nl.cv_r2(Fd, X),
        "cka_3digit": cka(Y, F3), "cka_3digit_shuf": cka(Y, F3[perm3]),
        "nn_adjacent": float((np.abs(x[nn] - x[:, None]) == 1).mean()),
        "nn_same_tens": float((x[nn] // 10 == x[:, None] // 10).mean()),
        "norm_mean_0_99": float(np.linalg.norm(E[:100], axis=1).mean()),
        "norm_mean_100_999": float(np.linalg.norm(E[100:1000], axis=1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--revisions", default="all")
    ap.add_argument("--cache", default=os.path.join(os.environ.get("HF_HOME", "/tmp"), "emb_shards"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    revs = pick_revisions() if args.revisions == "all" else args.revisions.split(",")
    tok = None
    rows = []
    done = out / "emb_manifold.jsonl"
    have = {json.loads(l)["revision"] for l in done.read_text().splitlines()} if done.exists() else set()
    for rev in revs:
        if rev in have:
            continue
        cache = Path(args.cache) / rev
        idx_path = hf_hub_download(REPO, "model.safetensors.index.json", revision=rev, local_dir=cache)
        wmap = json.loads(Path(idx_path).read_text())["weight_map"]
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(REPO, revision=rev)
            num_ids = nl.number_token_ids(tok, 999)
        mats = {}
        for key, name in (("model.embed_tokens.weight", "input"), ("lm_head.weight", "output"), ("model.norm.weight", "norm")):
            shard = hf_hub_download(REPO, wmap[key], revision=rev, local_dir=cache)
            with safe_open(shard, framework="pt") as f:
                t = f.get_tensor(key)
                mats[name] = (t[torch.tensor(num_ids)] if t.dim() == 2 else t).float().numpy()
        row = {"revision": rev, "input": metrics(mats["input"]), "output": metrics(mats["output"] * mats["norm"])}
        with done.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(rev, {k: round(v, 3) for k, v in row["input"].items()}, flush=True)
        shutil.rmtree(cache, ignore_errors=True)
    rows = [json.loads(l) for l in done.read_text().splitlines()]
    keys = ["cka_helix", "cka_digit", "cka_helix_shuf", "r2_helix", "r2_digit", "cka_3digit", "cka_3digit_shuf", "nn_adjacent", "nn_same_tens"]
    lines = ["# number manifold in the embedding matrices\n\n| revision | matrix | " + " | ".join(keys) + " |\n|---|---|" + "---|" * len(keys) + "\n"]
    for r in rows:
        for m in ("input", "output"):
            lines.append(f"| {r['revision']} | {m} | " + " | ".join(f"{r[m][k]:.2f}" for k in keys) + " |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
