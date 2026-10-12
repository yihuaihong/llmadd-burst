"""'Race' animation: held-out accuracy of the helix prior vs no prior during fine-tuning (7B 9B-token checkpoint,
7000 pairs, mean over seeds). Writes a 16:9 GIF (and MP4 if ffmpeg is available) plus the last frame as PNG.

    python tools/viz_race.py --out report/figures/race_9B results/b3-curve-9B-n7000-none-torch results/b3-curve-9B-n7000-helix-torch
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"helix": "#2a78d6", "none": "#eb6834"}          # categorical slots 1 and 2 (validated default palette)
LABEL = {"helix": "helix prior", "none": "no prior"}


def load(dirs):
    out = {}
    for d in dirs:
        rs = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/run_*_seed*.json"))]
        G = rs[0]["geom"]
        steps = np.array([c["step"] for c in rs[0]["curve"]])
        acc = 100 * np.mean([[c["test"] for c in r["curve"]] for r in rs], 0)
        out[G] = (np.concatenate([[0], steps]), np.concatenate([[0.0], acc]), len(rs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    data = load(args.dirs)
    steps = data["helix"][0]
    first = lambda G, thr: next((int(s) for s, a in zip(*data[G][:2]) if a >= thr), None)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, steps[-1] * 1.22); ax.set_ylim(0, 105)
    ax.set_xlabel("fine-tuning step", color=INK2); ax.set_ylabel("accuracy on unseen sums (%)", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    fig.text(0.07, 0.94, "A helix prior teaches an early LLM two-digit addition about 5x faster", fontsize=19, color=INK, weight="bold")
    fig.text(0.07, 0.895, f"OLMo-2-7B after 9B pretraining tokens, LoRA fine-tuning on 7000 sums; mean of {data['helix'][2]} seeds",
             fontsize=13, color=INK2)
    lines, dots, tags = {}, {}, {}
    for G in ("none", "helix"):
        lines[G], = ax.plot([], [], color=COLOR[G], linewidth=2.5, solid_capstyle="round")
        dots[G], = ax.plot([], [], "o", color=COLOR[G], markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        tags[G] = ax.text(0, 0, "", color=INK, fontsize=14, va="center")
    counter = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", color=INK2, fontsize=13)
    marks = []
    n = len(steps); hold = 25

    def frame(i):
        k = min(i, n - 1)
        for G in ("none", "helix"):
            s, a, _ = data[G]
            lines[G].set_data(s[:k + 1], a[:k + 1]); dots[G].set_data([s[k]], [a[k]])
            tags[G].set_position((s[k] + steps[-1] * 0.015, a[k])); tags[G].set_text(f"{LABEL[G]}  {a[k]:.0f}%")
        counter.set_text(f"step {int(steps[k])}" if k < n - 1 else "")
        if k == n - 1 and not marks:
            xh, xn = first("helix", 50), first("none", 50)
            if xh is not None and xn is not None:
                for G, x in (("helix", xh), ("none", xn)):
                    marks.append(ax.plot([x, x], [0, 50], color=COLOR[G], linewidth=1, linestyle=(0, (3, 3)))[0])
                marks.append(ax.annotate("", xy=(xn, 50), xytext=(xh, 50),
                                         arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.5, shrinkA=0, shrinkB=0)))
                marks.append(ax.text((xh + xn) / 2, 52.5, f"{xn / xh:.0f}x fewer steps to reach 50%  ({xh} vs {xn})",
                                     ha="center", va="bottom", color=INK, fontsize=13))
            counter.set_text("")
        return [*lines.values(), *dots.values(), *tags.values(), counter, *marks]

    anim = FuncAnimation(fig, frame, frames=n + hold, interval=80, blit=False)
    anim.save(f"{args.out}.gif", writer=PillowWriter(fps=12))
    if shutil.which("ffmpeg"):
        anim.save(f"{args.out}.mp4", writer=FFMpegWriter(fps=12, bitrate=2400))
    frame(n + hold - 1)
    fig.savefig(f"{args.out}.png", facecolor=SURFACE)
    print("wrote", args.out, {G: first(G, 50) for G in data})


if __name__ == "__main__":
    main()
