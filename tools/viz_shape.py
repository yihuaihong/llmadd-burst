"""'Shape' animation: the layer-16 states of the numbers 10..99 during fine-tuning, with vs without the helix prior
(snapshots from manifold_train.py --save_states_every). Each frame: the mean state of each units digit 0..9 (over
the tens), 2-D PCA of those ten means (no target is fitted), rotated onto the arm's final frame (orthogonal
Procrustes) so the view does not flip; the digits are joined 0 -> 1 -> ... -> 9 -> 0, and the 90 numbers are drawn
faintly in the same projection. Tight clusters = numbers that share a last digit are represented alike. Side by side: no prior | helix.

    python tools/viz_shape.py --out report/figures/shape_9B --none <dir> --helix <dir>
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
from matplotlib.patches import Ellipse

SURFACE, INK, INK2, LINE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c3", "#e6e5e1"
ARM_COLOR = {"no prior": "#eb6834", "helix prior": "#2a78d6"}   # same slots as the race figure


def eta2(H):
    """Share of the (full-dimensional) variance of the 90 number states explained by the last digit."""
    nums = np.arange(10, 100)
    mu = H.mean(0)
    between = sum(9 * np.sum((H[nums % 10 == d].mean(0) - mu) ** 2) for d in range(10))
    return float(between / np.sum((H - mu) ** 2))


def digit_view(H):
    """(10 digit means, 90 numbers) in the 2-D PCA plane of the digit means, scaled to unit radius."""
    nums = np.arange(10, 100)
    M = np.stack([H[nums % 10 == d].mean(0) for d in range(10)])
    mu = M.mean(0)
    _, _, Vt = np.linalg.svd(M - mu, full_matrices=False)
    P = Vt[:2].T
    m, x = (M - mu) @ P, (H - mu) @ P
    r = np.abs(m).max() + 1e-9
    return m / r, x / r


def procrustes(X, Y):
    """Rotate X onto Y (orthogonal, no scaling)."""
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return X @ (U @ Vt)


def load(d):
    z = np.load(glob.glob(f"{d}/states_*_seed0.npz")[0])
    r = json.load(open(glob.glob(f"{d}/run_*_seed0.json")[0]))
    acc = {c["step"]: c["test"] for c in r["curve"]}
    Hs = [H.astype(np.float32) for H in z["states"]]
    V = [digit_view(H) for H in Hs]
    eta = [eta2(H) for H in Hs]
    out = []
    for m, x in V:
        U, _, Wt = np.linalg.svd(m.T @ V[-1][0]); R = U @ Wt
        out.append((m @ R, x @ R))
    return z["steps"], out, acc, eta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--none", required=True)
    ap.add_argument("--helix", required=True)
    args = ap.parse_args()
    arms = {"no prior": load(args.none), "helix prior": load(args.helix)}
    n = min(len(v[0]) for v in arms.values()); hold = 20
    tmax = int(max(v[0][n - 1] for v in arms.values()))
    nums = np.arange(10, 100)
    cmap = plt.get_cmap("twilight")
    dcol = cmap(0.08 + 0.84 * np.arange(10) / 9.0)   # skip the near-white ends of the cyclic map
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.05, 0.93, "With the helix prior, numbers that share a last digit snap together", fontsize=19, color=INK, weight="bold")
    fig.text(0.05, 0.885, "Big dots: mean state of each last digit; small dots: the numbers 10-99 (OLMo-2-7B, 9B tokens, layer 16, plain PCA)",
             fontsize=12, color=INK2)
    fig.text(0.05, 0.855, "Shaded ellipses: 2-sigma spread of each last-digit group; lines join each number to its group centre",
             fontsize=12, color=INK2)
    axes = {}
    for i, name in enumerate(arms):
        ax = fig.add_axes([0.06 + 0.47 * i, 0.27, 0.42, 0.55])
        ax.set_facecolor(SURFACE)
        ax.set_axis_off()
        axes[name] = ax
    strip = fig.add_axes([0.1, 0.07, 0.82, 0.14])

    def frame(f):
        k = min(f, n - 1)
        strip.cla(); strip.set_facecolor(SURFACE)
        for name, (steps, X, acc, eta) in arms.items():
            strip.plot(steps[:k + 1], 100 * np.array(eta[:k + 1]), color=ARM_COLOR[name], linewidth=2.5)
            strip.plot([steps[k]], [100 * eta[k]], "o", color=ARM_COLOR[name], markersize=7, markeredgecolor=SURFACE)
            strip.text(steps[k] + tmax * 0.01, 100 * eta[k], f"{name} {100 * eta[k]:.0f}%", color=INK, fontsize=11, va="center")
        strip.set_xlim(0, tmax * 1.18); strip.set_ylim(0, 100)
        strip.set_ylabel("variance explained\nby last digit (%)", color=INK2, fontsize=10)
        strip.set_xlabel("fine-tuning step", color=INK2, fontsize=10)
        for sp in ("top", "right"):
            strip.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            strip.spines[sp].set_color(INK2)
        strip.tick_params(colors=INK2, labelsize=10); strip.yaxis.grid(True, color=GRID); strip.set_axisbelow(True)
        for name, (steps, X, acc, eta) in arms.items():
            ax = axes[name]; ax.cla(); ax.set_axis_off(); ax.set_facecolor(SURFACE)
            m, x = X[k]
            for d in range(10):   # 2-sigma ellipse of each last-digit cluster + spokes to its centre
                pts = x[nums % 10 == d]
                C = np.cov(pts.T) + 1e-9 * np.eye(2)
                w, v = np.linalg.eigh(C)
                ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
                ax.add_patch(Ellipse(pts.mean(0), 4 * np.sqrt(w[-1]), 4 * np.sqrt(w[0]), angle=ang,
                                     facecolor=dcol[d], alpha=0.12, edgecolor=dcol[d], linewidth=1))
                for q in pts:
                    ax.plot([q[0], m[d, 0]], [q[1], m[d, 1]], color=dcol[d], alpha=0.25, linewidth=0.6)
            ax.scatter(x[:, 0], x[:, 1], c=dcol[nums % 10], s=22, alpha=0.6, linewidths=0)
            ring = np.vstack([m, m[:1]])

            ax.scatter(m[:, 0], m[:, 1], c=dcol, s=260, edgecolors=SURFACE, linewidths=2, zorder=3)
            for d in range(10):
                ax.text(m[d, 0], m[d, 1], str(d), ha="center", va="center", fontsize=11, color="white", weight="bold", zorder=4)
            ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal")
            a = acc.get(int(steps[k]))
            ax.set_title(f"{name}  ·  step {int(steps[k])}" + (f"  ·  unseen sums {100 * a:.0f}%" if a is not None else ""),
                         color=INK, fontsize=14, pad=2)
        return []

    anim = FuncAnimation(fig, frame, frames=n + hold, interval=100)
    anim.save(f"{args.out}.gif", writer=PillowWriter(fps=10))
    frame(n + hold - 1)
    fig.savefig(f"{args.out}.png", facecolor=SURFACE)
    print("wrote", args.out, "frames", n)


if __name__ == "__main__":
    main()
