"""'Helix' animation: the layer-16 states of the numbers 10..99 during fine-tuning, with vs without the helix prior,
drawn in 3-D axes computed from the states themselves (nothing is trained or fitted to a target):
  horizontal plane = the period-10 Fourier component of the ten last-digit means
                     (u_c = sum_d m_d cos(2 pi d / 10), u_s = sum_d m_d sin(2 pi d / 10), orthonormalised)
  height           = the first principal direction of the nine tens-digit means, orthogonalised to that plane
If the last digit sits on a circle and magnitude runs along one axis, the line 10 -> 11 -> ... -> 99 winds up as a
helix (one turn per ten numbers). Points coloured by the last digit; the camera turns slowly; a strip at the bottom
shows the linear CKA between the states and the ideal helix (the same quantity the prior optimises).

    python tools/viz_helix3d.py --out report/figures/helix3d_9B --none <dir> --helix <dir>
"""

from __future__ import annotations

import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import numlib as nl

SURFACE, INK, INK2, LINE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e89", "#e6e5e1"
ARM_COLOR = {"no prior": "#eb6834", "helix prior": "#2a78d6"}
NUMS = np.arange(10, 100)


def axes_of(H):
    mu = H.mean(0)
    M = np.stack([H[NUMS % 10 == d].mean(0) for d in range(10)]) - mu
    th = 2 * np.pi * np.arange(10) / 10
    uc, us = np.cos(th) @ M, np.sin(th) @ M
    e1 = uc / (np.linalg.norm(uc) + 1e-9)
    us = us - (us @ e1) * e1
    e2 = us / (np.linalg.norm(us) + 1e-9)
    T = np.stack([H[NUMS // 10 == t].mean(0) for t in range(1, 10)]) - mu
    T = T - np.outer(T @ e1, e1) - np.outer(T @ e2, e2)
    e3 = np.linalg.svd(T, full_matrices=False)[2][0]
    if (T @ e3)[-1] < (T @ e3)[0]:
        e3 = -e3
    X = (H - mu) @ np.stack([e1, e2, e3], 1)
    return X, M @ np.stack([e1, e2], 1)


def circularity(md):
    """How well the ten last-digit means sit on an ordered circle in the plane: correlation between their angles
    and the ideal angles 2 pi d / 10 (circular), times the share of their spread that lies in the plane radius."""
    ang = np.arctan2(md[:, 1], md[:, 0])
    ideal = 2 * np.pi * np.arange(10) / 10
    return float(np.abs(np.mean(np.exp(1j * (ang - ideal)))))


def cka_helix(H):
    F = nl.features(NUMS)[0]
    F = (F - F.mean(0)) / F.std(0)
    Hc = H - H.mean(0)
    G, K = Hc @ Hc.T, F @ F.T
    K = K - K.mean(0) - K.mean(1)[:, None] + K.mean()
    return float((G * K).sum() / (np.linalg.norm(G) * np.linalg.norm(K)))


def load(d):
    z = np.load(glob.glob(f"{d}/states_*_seed0.npz")[0])
    r = json.load(open(glob.glob(f"{d}/run_*_seed0.json")[0]))
    acc = {c["step"]: c["test"] for c in r["curve"]}
    frames, circ = [], []
    for H in z["states"]:
        X, md = axes_of(H.astype(np.float32))
        rad = np.linalg.norm(md, axis=1).mean() + 1e-9
        X[:, :2] /= rad
        X[:, 2] = 2 * (X[:, 2] - X[:, 2].min()) / (np.ptp(X[:, 2]) + 1e-9) - 1
        frames.append(X); circ.append(cka_helix(H.astype(np.float64)))
    return z["steps"], frames, acc, circ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--none", required=True)
    ap.add_argument("--helix", required=True)
    args = ap.parse_args()
    arms = {"no prior": load(args.none), "helix prior": load(args.helix)}
    n = min(len(v[0]) for v in arms.values()); hold = 30
    tmax = int(max(v[0][n - 1] for v in arms.values()))
    cmap = plt.get_cmap("twilight")
    col = cmap(0.08 + 0.84 * (NUMS % 10) / 9.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.05, 0.94, "The helix prior turns the model's rough coil of numbers into a clean helix", fontsize=19, color=INK, weight="bold")
    fig.text(0.05, 0.905, "Numbers 10 -> 99 joined in order (layer 16, OLMo-2-7B at 9B tokens). Plane: period-10 part of the "
             "last-digit means; height: tens. Colour = last digit.", fontsize=10.5, color=INK2)
    axes = {name: fig.add_axes([0.02 + 0.49 * i, 0.22, 0.47, 0.6], projection="3d") for i, name in enumerate(arms)}
    strip = fig.add_axes([0.1, 0.06, 0.82, 0.12])

    def frame(f):
        k = min(f, n - 1)
        strip.cla(); strip.set_facecolor(SURFACE)
        for name, (steps, X, acc, circ) in arms.items():
            strip.plot(steps[:k + 1], circ[:k + 1], color=ARM_COLOR[name], linewidth=2.5)
            strip.plot([steps[k]], [circ[k]], "o", color=ARM_COLOR[name], markersize=7, markeredgecolor=SURFACE)
            strip.text(steps[k] + tmax * 0.01, circ[k], f"{name} {circ[k]:.2f}", color=INK, fontsize=11, va="center")
        strip.set_xlim(0, tmax * 1.18); strip.set_ylim(0.5, 1.02)
        strip.set_ylabel("similarity to an\nideal helix (CKA)", color=INK2, fontsize=10)
        strip.set_xlabel("fine-tuning step", color=INK2, fontsize=10)
        for sp in ("top", "right"):
            strip.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            strip.spines[sp].set_color(INK2)
        strip.tick_params(colors=INK2, labelsize=10); strip.yaxis.grid(True, color=GRID); strip.set_axisbelow(True)
        for name, (steps, X, acc, circ) in arms.items():
            ax = axes[name]; ax.cla(); ax.set_facecolor(SURFACE)
            x = X[k]
            ax.plot(x[:, 0], x[:, 1], x[:, 2], color=LINE, linewidth=1.0, alpha=0.8)
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=col, s=30, depthshade=False, edgecolors=SURFACE, linewidths=0.5)
            ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(-1.1, 1.1)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_facecolor(SURFACE); pane.set_edgecolor(GRID)
            ax.view_init(elev=18, azim=-60 + 1.2 * f)
            a = acc.get(int(steps[k]))
            ax.set_title(f"{name}  ·  step {int(steps[k])}" + (f"  ·  unseen sums {100 * a:.0f}%" if a is not None else ""),
                         color=INK, fontsize=13, pad=-4)
        return []

    anim = FuncAnimation(fig, frame, frames=n + hold, interval=100)
    anim.save(f"{args.out}.gif", writer=PillowWriter(fps=10))
    frame(n + hold - 1)
    fig.savefig(f"{args.out}.png", facecolor=SURFACE)
    print("wrote", args.out, "frames", n, {k: round(v[3][-1], 2) for k, v in arms.items()})


if __name__ == "__main__":
    main()
