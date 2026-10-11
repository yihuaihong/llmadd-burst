"""E6: is there a benefit window for a geometry prior in a toy model that learns modular addition from scratch?

Setting of Nanda et al. 2023 (progress measures for grokking): (a + b) mod p, p = 113, 30% of the pairs for
training, one-layer transformer (d 128, 4 heads, MLP 512, no LayerNorm), AdamW lr 1e-3, weight decay 1, full batch.
Phases without any prior: memorisation, circuit formation, cleanup (test accuracy jumps).

The geometry loss is the LLM one (1 - linear CKA between the number-token embeddings W_E[0:p] and an ideal circle
cos/sin(2 pi a / p)), switched on only for steps [t0, t0 + width): does the prior speed up grokking at every time,
or only in one phase? 'always' = on for the whole run; '_shuf' = the circle with permuted numbers.

    python tools/toy_grok.py --out <dir> --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class OneLayer(nn.Module):
    def __init__(self, p: int, d: int = 128, heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / d ** 0.5)
        self.W_pos = nn.Parameter(torch.randn(3, d) / d ** 0.5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp_in, self.mlp_out = nn.Linear(d, d_mlp), nn.Linear(d_mlp, d)
        self.W_U = nn.Linear(d, p, bias=False)

    def forward(self, x):
        h = self.W_E[x] + self.W_pos
        B, T, D = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.o(att.transpose(1, 2).reshape(B, T, D))
        h = h + self.mlp_out(Fn.relu(self.mlp_in(h)))
        return self.W_U(h[:, -1])


def centred_gram(F: torch.Tensor) -> torch.Tensor:
    F = F - F.mean(0)
    return F @ F.T


def cka(H: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    G = centred_gram(H)
    return (G * K).sum() / (G.norm() * K.norm() + 1e-12)


def fourier_top(W: torch.Tensor, k: int = 5) -> float:
    """Share of the embedding's (non-constant) Fourier power in its top-k frequencies (cos+sin pooled)."""
    P = torch.fft.rfft(W.float(), dim=0).abs().pow(2).sum(1)[1:]
    return float(P.topk(k).values.sum() / P.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--arms", default="none,always,w0,w1000,w2000,w4000,w6000,w8000,w12000,always_shuf,w2000_shuf,w6000_shuf")
    ap.add_argument("--eval_every", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = args.p
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    X_all = torch.tensor(np.stack([a.ravel(), b.ravel(), np.full(p * p, p)], 1), device=dev)
    Y_all = (X_all[:, 0] + X_all[:, 1]) % p
    ang = 2 * np.pi * np.arange(p) / p
    circ = torch.tensor(np.stack([np.cos(ang), np.sin(ang)], 1), dtype=torch.float32, device=dev)
    K = {"circle": centred_gram(circ),
         "circle_shuf": centred_gram(circ[torch.tensor(np.random.default_rng(5).permutation(p), device=dev)])}
    t_start = time.time()
    for seed in [int(s) for s in args.seeds.split(",")]:
        perm = torch.tensor(np.random.default_rng(seed).permutation(p * p), device=dev)
        ntr = int(args.frac * p * p)
        tr, te = perm[:ntr], perm[ntr:]
        for arm in args.arms.split(","):
            path = out / f"run_{arm}_seed{seed}.json"
            if path.exists():
                continue
            target = "circle_shuf" if arm.endswith("_shuf") else "circle"
            base = arm[: -len("_shuf")] if arm.endswith("_shuf") else arm
            on = (lambda s: False) if base == "none" else (lambda s: True) if base == "always" else \
                (lambda s, t0=int(base[1:]): t0 <= s < t0 + args.width)
            torch.manual_seed(seed)
            model = OneLayer(p).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
            curve = []
            for s in range(args.steps):
                for g in opt.param_groups:
                    g["lr"] = 1e-3 * min(1.0, (s + 1) / 10)
                ce = Fn.cross_entropy(model(X_all[tr]), Y_all[tr])
                loss = ce + (args.lam * (1 - cka(model.W_E[:p], K[target])) if on(s) else 0.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if s % args.eval_every == 0 or s == args.steps - 1:
                    with torch.no_grad():
                        acc = lambda idx: float((model(X_all[idx]).argmax(-1) == Y_all[idx]).float().mean())
                        curve.append({"step": s, "ce": float(ce), "train": acc(tr), "test": acc(te),
                                      "cka_circle": float(cka(model.W_E[:p], K["circle"])), "fourier_top5": fourier_top(model.W_E[:p])})
            first = lambda key, thr: next((c["step"] for c in curve if c[key] >= thr), None)
            res = {"arm": arm, "seed": seed, "lam": args.lam, "width": args.width, "p": p, "frac": args.frac,
                   "t_mem": first("train", 0.99), "t_grok": first("test", 0.95), "final_test": curve[-1]["test"], "curve": curve}
            path.write_text(json.dumps(res))
            print(f"{arm} seed{seed}: t_mem {res['t_mem']} t_grok {res['t_grok']} final test {res['final_test']:.3f} ({time.time() - t_start:.0f}s)", flush=True)
    summarize(out)


def summarize(out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    arms = sorted({r["arm"] for r in runs}, key=lambda a: (a.endswith("_shuf"), a != "none", a != "always",
                                                            int(a.split("_")[0][1:]) if a.startswith("w") else 0))
    lines = ["# toy grokking: geometry prior switched on in a window (p, frac, lam, width in the run files)\n\n",
             "| arm | seeds | t_mem | t_grok (mean ± sd; ungrokked = never) | final test |\n|---|---|---|---|---|\n"]
    for A in arms:
        rs = [r for r in runs if r["arm"] == A]
        tg = [r["t_grok"] for r in rs if r["t_grok"] is not None]
        tm = [r["t_mem"] for r in rs if r["t_mem"] is not None]
        g = f"{np.mean(tg):.0f} ± {np.std(tg):.0f} ({len(tg)}/{len(rs)})" if tg else f"never (0/{len(rs)})"
        lines.append(f"| {A} | {len(rs)} | {np.mean(tm):.0f} | {g} | {np.mean([r['final_test'] for r in rs]):.3f} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
