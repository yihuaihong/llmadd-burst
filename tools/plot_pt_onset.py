"""When to switch the helix prior on in from-scratch pretraining: held-out accuracy (training position) for windows that
start at different points and end at 40% of training, against no prior and the whole-run prior. One sequential colour
scale for the start time; mean over seeds with the seed range shaded; dotted verticals mark each window's start.

    python tools/plot_pt_onset.py --out report/figures/pt_onset results/ptstart-torch results/pt7c-torch [--key test_ctx]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="test_ctx")
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    groups = {}
    for d in args.dirs:
        for p in glob.glob(f"{d}/run_*.json") + glob.glob(f"{d}/*/run_*.json"):
            r = json.load(open(p)); a = r["args"]
            if a.get("geo_site", "prefix") != "prefix" or a.get("split", "ordered") != "ordered":
                continue
            if a["arm"] == "none":
                g = "none"
            elif a["arm"] == "helix":
                g = "whole"
            elif a["arm"] == "helix_window" and a["window"].endswith(",0.4") and not a.get("lam_after"):
                g = float(a["window"].split(",")[0])
            else:
                continue
            groups.setdefault(g, {})[a["seed"]] = r   # one run per seed
    starts = sorted(k for k in groups if isinstance(k, float))
    cmap = plt.get_cmap("viridis")
    colour = {s: cmap(0.15 + 0.7 * i / max(1, len(starts) - 1)) for i, s in enumerate(starts)}
    colour.update({"none": "#eb6834", "whole": "#8f8e89"})
    label = {s: f"helix from {int(100 * s)}% to 40% of training" for s in starts}
    label.update({"none": "no prior", "whole": "helix, whole run"})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    T = None
    for g in ["none", "whole", *starts]:
        rs = list(groups.get(g, {}).values())
        if not rs:
            continue
        n = max(len(r["curve"]) for r in rs)
        A = np.full((len(rs), n), np.nan)
        for i, r in enumerate(rs):
            A[i, :len(r["curve"])] = [100 * e[args.key] for e in r["curve"]]
        tok = np.array([e["tokens"] for e in max(rs, key=lambda r: len(r["curve"]))["curve"]]) / 1e6
        T = rs[0]["args"]["tokens"] / 1e6
        ax.fill_between(tok, np.nanmin(A, 0), np.nanmax(A, 0), color=colour[g], alpha=0.12, linewidth=0)
        ax.plot(tok, np.nanmean(A, 0), color=colour[g], linewidth=2.6 if g not in ("none", "whole") else 2.2,
                label=f"{label[g]} (n={len(rs)})")
        if isinstance(g, float) and g > 0:
            ax.axvline(g * T, color=colour[g], linewidth=1, linestyle=(0, (2, 3)))
    ax.set_xlabel("pretraining tokens (millions)", color=INK2); ax.set_ylabel("held-out sums correct (%)", color=INK2)
    ax.set_ylim(0, 102); ax.set_xlim(0, T)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    ax.set_title("When to switch the helix prior on (53M-param model pretrained from scratch, ~7% arithmetic)",
                 color=INK, fontsize=12.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", facecolor=SURFACE)
    print("wrote", f"{args.out}.png", {str(g): len(v) for g, v in groups.items()})


if __name__ == "__main__":
    main()
