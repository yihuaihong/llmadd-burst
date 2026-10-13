"""Held-out addition accuracy during from-scratch pretraining, measured in training position (*_ctx keys of
pretrain_helix.py --ctx_eval), mean over seeds with the seed range shaded; one line per CONDITION (arm + window /
gate + geometry site, as in tools/pt_summary.py). Curves of different lengths are not truncated (NaN-padded).

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

from pt_summary import label

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"helix_window": "#2a78d6", "helix": "#1baf7a", "none": "#eb6834", "helix_shuf": "#8f8e89"}
LABEL = {"helix_window": "helix prior, 10-40% of training", "helix": "helix prior, whole run", "none": "no prior",
         "helix_shuf": "shuffled helix, whole run"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=None, help="default: uheld_ctx for --split unordered runs, else test_ctx")
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    runs = {}
    for d in args.dirs:
        for p in sorted(set(glob.glob(f"{d}/run_*.json") + glob.glob(f"{d}/*/run_*.json"))):
            r = json.load(open(p)); runs.setdefault(label(r["args"]), []).append(r)
    first = next(iter(runs.values()))[0]
    key = args.key or ("uheld_ctx" if first["args"].get("split") == "unordered" else "test_ctx")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    extra = iter(plt.get_cmap("tab10").colors)
    for lab in sorted(runs, key=lambda k: (k != "none", k)):
        rs = runs[lab]
        n = max(len(r["curve"]) for r in rs)
        if any(len(r["curve"]) != n for r in rs):
            print(f"warning: {lab}: curves of different lengths {[len(r['curve']) for r in rs]} (NaN-padded)")
        tok = np.array([e["tokens"] for e in max(rs, key=lambda r: len(r["curve"]))["curve"]]) / 1e6
        A = np.full((len(rs), n), np.nan)
        for i, r in enumerate(rs):
            A[i, :len(r["curve"])] = [100 * e[key] for e in r["curve"]]
        arm = rs[0]["args"]["arm"]
        name = LABEL.get(arm, arm) if lab == arm or (arm == "helix_window" and lab == "helix_window 0.1,0.4") else lab
        col = COLOR.get(arm) if name == LABEL.get(arm) else next(extra)
        ax.fill_between(tok, np.nanmin(A, 0), np.nanmax(A, 0), color=col, alpha=0.15, linewidth=0)
        ax.plot(tok, np.nanmean(A, 0), color=col, linewidth=2.4, label=f"{name} (n={len(rs)})")
        if arm == "helix_window" and len(runs) <= 5:
            w0, w1 = (float(x) for x in rs[0]["args"]["window"].split(","))
            T = rs[0]["args"]["tokens"] / 1e6
            ax.axvspan(w0 * T, w1 * T, color=col, alpha=0.06, linewidth=0)
    ax.set_xlabel("pretraining tokens (millions)", color=INK2)
    ax.set_ylabel("held-out sums correct (%)" + ("\n(both orders unseen)" if key.startswith("uheld") else ""), color=INK2)
    ax.set_ylim(0, 102); ax.set_xlim(0, None)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    lines = first["args"].get("arith_lines") or 1
    share = {1: "~2%", 4: "~7%"}.get(lines, f"{lines} lines")
    ax.set_title(f"From-scratch pretraining (53M params, {share} arithmetic): held-out two-digit sums, in training position",
                 color=INK, fontsize=12.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", facecolor=SURFACE)
    print("wrote", f"{args.out}.png")


if __name__ == "__main__":
    main()
