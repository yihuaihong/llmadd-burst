"""Held-out addition accuracy during from-scratch pretraining, measured in training position (*_ctx keys of
pretrain_helix.py --ctx_eval), mean over seeds with the seed range shaded; one line per arm.

    python tools/plot_pt_curves.py --out report/figures/pt7c_curves results/pt7c-torch [--key test_ctx]
"""

from __future__ import annotations

import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"helix_window": "#2a78d6", "helix": "#1baf7a", "none": "#eb6834", "helix_shuf": "#8f8e89"}
LABEL = {"helix_window": "helix prior, 10-40% of training", "helix": "helix prior, whole run", "none": "no prior",
         "helix_shuf": "shuffled helix, whole run"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="test_ctx")
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    runs = {}
    for d in args.dirs:
        for p in glob.glob(f"{d}/run_*.json"):
            r = json.load(open(p)); runs.setdefault(r["args"]["arm"], []).append(r)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for arm in ("none", "helix_shuf", "helix", "helix_window"):
        if arm not in runs:
            continue
        rs = runs[arm]
        n = min(len(r["curve"]) for r in rs)
        tok = np.array([e["tokens"] for e in rs[0]["curve"][:n]]) / 1e6
        A = 100 * np.array([[e[args.key] for e in r["curve"][:n]] for r in rs])
        ax.fill_between(tok, A.min(0), A.max(0), color=COLOR[arm], alpha=0.15, linewidth=0)
        ax.plot(tok, A.mean(0), color=COLOR[arm], linewidth=2.4, label=f"{LABEL[arm]} (n={len(rs)})")
    a = runs.get("helix_window", [{}])[0].get("args", {})
    if a.get("window"):
        w0, w1 = (float(x) for x in a["window"].split(","))
        T = a["tokens"] / 1e6
        ax.axvspan(w0 * T, w1 * T, color=COLOR["helix_window"], alpha=0.06, linewidth=0)
        ax.text((w0 + w1) / 2 * T, 3, "window prior on", ha="center", color=INK2, fontsize=11)
    ax.set_xlabel("pretraining tokens (millions)", color=INK2); ax.set_ylabel("held-out sums correct (%)", color=INK2)
    ax.set_ylim(0, 102); ax.set_xlim(0, None)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    lines = next(iter(runs.values()))[0]["args"].get("arith_lines") or 1
    share = {1: "~2%", 4: "~7%"}.get(lines, f"{lines} lines")
    ax.set_title(f"From-scratch pretraining (53M params, {share} arithmetic): held-out two-digit sums, in training position",
                 color=INK, fontsize=12.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", facecolor=SURFACE)
    print("wrote", f"{args.out}.png")


if __name__ == "__main__":
    main()
