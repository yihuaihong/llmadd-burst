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

SURFACE, INK, INK2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c3"


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
    V = [digit_view(H.astype(np.float32)) for H in z["states"]]
    out = []
    for m, x in V:
        U, _, Wt = np.linalg.svd(m.T @ V[-1][0]); R = U @ Wt
        out.append((m @ R, x @ R))
    return z["steps"], out, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--none", required=True)
    ap.add_argument("--helix", required=True)
    args = ap.parse_args()
    arms = {"no prior": load(args.none), "helix prior": load(args.helix)}
    n = min(len(v[0]) for v in arms.values()); hold = 20
    nums = np.arange(10, 100)
    cmap = plt.get_cmap("twilight")
    dcol = cmap(0.08 + 0.84 * np.arange(10) / 9.0)   # skip the near-white ends of the cyclic map
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.05, 0.93, "With the helix prior, numbers that share a last digit snap together", fontsize=19, color=INK, weight="bold")
    fig.text(0.05, 0.885, "Big dots: mean state of each last digit; small dots: the numbers 10-99 (OLMo-2-7B, 9B tokens, layer 16, plain PCA)",
             fontsize=12, color=INK2)
    axes = {}
    for i, name in enumerate(arms):
        ax = fig.add_axes([0.06 + 0.47 * i, 0.06, 0.42, 0.72])
        ax.set_facecolor(SURFACE)
        ax.set_axis_off()
        axes[name] = ax

    def frame(f):
        k = min(f, n - 1)
        for name, (steps, X, acc) in arms.items():
            ax = axes[name]; ax.cla(); ax.set_axis_off(); ax.set_facecolor(SURFACE)
            m, x = X[k]
            ax.scatter(x[:, 0], x[:, 1], c=dcol[nums % 10], s=22, alpha=0.55, linewidths=0)
            ring = np.vstack([m, m[:1]])

            ax.scatter(m[:, 0], m[:, 1], c=dcol, s=260, edgecolors=SURFACE, linewidths=2, zorder=3)
            for d in range(10):
                ax.text(m[d, 0], m[d, 1], str(d), ha="center", va="center", fontsize=11, color="white", weight="bold", zorder=4)
            ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal")
            a = acc.get(int(steps[k]))
            ax.set_title(f"{name}\nstep {int(steps[k])}" + (f"  ·  unseen sums {100 * a:.0f}%" if a is not None else ""),
                         color=INK, fontsize=14, pad=0)
        return []

    anim = FuncAnimation(fig, frame, frames=n + hold, interval=100)
    anim.save(f"{args.out}.gif", writer=PillowWriter(fps=10))
    frame(n + hold - 1)
    fig.savefig(f"{args.out}.png", facecolor=SURFACE)
    print("wrote", args.out, "frames", n)


if __name__ == "__main__":
    main()
