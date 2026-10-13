"""Figures for the manifold atlas (tools/manifold_atlas.py outputs atlas_<tag>.json / states_<tag>.npz):
  <out>_timeline.png  per domain, how decodable its geometry is (LOO R^2 of the primary target minus the permutation
                      null, best layer) and its 3-shot test-set accuracy (rescaled so chance = 0, perfect = 1), across
                      pretraining checkpoints; one colour per domain (rainbow order by kind)
  <out>_overlap.png   at one checkpoint: share of domain b's geometry that lives in domain a's subspace,
                      1 - R^2(b without a's basis) / R^2(b), at the layer where b is best decoded
  <out>_gallery.png   at one checkpoint: each domain's leave-one-out predicted coordinates (honest, not in-sample),
                      coloured by the true coordinate

  <out>_families.png  (atlas v2, --families) per domain × model family (final checkpoints): geometry score and test
                      accuracy, to see which manifolds are universal

    python tools/viz_atlas.py --dir <atlas out dir> --out report/figures/atlas [--tag 210B]
    atlas v2 (tags "<model>-<tokens>"): --prefix olmo2-7b- selects one trajectory; --families olmo2-7b-main,qwen2.5-7b,...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manifold_domains import domains

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
KIND_ORDER = ("helix", "cyclic", "linear", "magnitude", "kin", "scalar", "tree", "sphere", "plane")


def tokens_of(tag):
    tag = tag.rsplit("-", 1)[-1]   # v2 tags: "<model>-<tokens>"
    if tag == "main":
        return 4.05e12
    if tag in ("rand", "0B"):   # random-init controls, drawn left of the first checkpoint
        return 2e8
    m = re.match(r"(\d+)B", tag)
    return float(m.group(1)) * 1e9 if m else np.nan


def load(d, prefix=None):
    res = {}
    for p in Path(d).glob("atlas_*.json"):
        r = json.loads(p.read_text())
        if prefix is None or r["tag"].startswith(prefix):
            res[r["tag"]] = r
    return dict(sorted(res.items(), key=lambda kv: tokens_of(kv[0])))


def order(r):
    dm = r["domains"]
    return sorted(dm, key=lambda k: (KIND_ORDER.index(dm[k]["kind"]), k))


def best(fits, key="r2"):
    v = [f[key] - f["null"] for f in fits]
    i = int(np.argmax(v))
    return v[i], i


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)


def geo_score(r, tmpl, k, prim):
    """Best layer R^2 minus the best-layer permutation null (no layer-selection bonus)."""
    b = r[tmpl]["best"][k][prim]
    return b["r2"] - b["null"]


def token_confound(r, k, prim, thr=0.3):
    c = r.get("token_sum_random", {}).get(k, {}).get(prim)
    return c is not None and c["r2"] - c["null"] >= thr


def timeline(res, out, tmpl):
    tags = list(res)
    names = order(res[tags[-1]])
    cols = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(names)))
    x = np.array([tokens_of(t) for t in tags])
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for c, k in zip(cols, names):
        prim = res[tags[-1]]["domains"][k]["primary"]
        geo = [geo_score(res[t], tmpl, k, prim) for t in tags]
        k_lab = k + (" *" if token_confound(res[tags[-1]], k, prim) else "")
        tk = [(res[t]["tasks"][k]["acc"] - res[t]["tasks"][k]["chance"]) / (1 - res[t]["tasks"][k]["chance"]) for t in tags]
        axs[0].plot(x, geo, "-o", color=c, linewidth=2, markersize=4)
        axs[0].text(x[-1] * 1.15, geo[-1], k_lab, color=c, fontsize=9, va="center")
        axs[1].plot(x, tk, "-o", color=c, linewidth=2, markersize=4)
        axs[1].text(x[-1] * 1.15, tk[-1], k, color=c, fontsize=9, va="center")
    for ax, t in zip(axs, ("geometry decodable from the states (best layer LOO R² − best-layer null)\n* = also decodable from the tokenisation alone (random-embedding control)",
                          "3-shot test set\n(accuracy rescaled: chance = 0, perfect = 1)")):
        style(ax); ax.set_xscale("log"); ax.set_xlabel("pretraining tokens (OLMo-2-7B checkpoints)", color=INK2)
        ax.set_title(t, color=INK, fontsize=11.5, loc="left"); ax.set_xlim(x[0] / 1.5, x[-1] * 4)
        if "rand" in tags:
            ax.axvline(x[tags.index("rand")] * 1.6, color=GRID, linewidth=1)
            ax.text(x[tags.index("rand")], ax.get_ylim()[1] * 0.97, "random\ninit", fontsize=8, color=INK2, ha="center", va="top")
    fig.suptitle(f"When does each concept manifold form? ({tmpl} prompts)", color=INK, fontsize=15, x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}_timeline.png", facecolor=SURFACE)


def overlap(res, out, tag, tmpl):
    r = res[tag][tmpl]
    names = r["order"]; idx = {k: i for i, k in enumerate(names)}
    show = order(res[tag])
    M = np.zeros((len(show), len(show)))
    for j, b in enumerate(show):
        prim = res[tag]["domains"][b]["primary"]
        li = res[tag]["layers"].index(r["best"][b][prim]["layer"])
        full = r["best"][b][prim]["r2"]
        for i, a in enumerate(show):
            rem = np.array(r["removal"][li])[idx[a], idx[b]]
            M[i, j] = 1 - rem / full if full > 0.05 else np.nan
    fig, ax = plt.subplots(figsize=(10.5, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(np.clip(M, 0, 1), cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=60, ha="right", fontsize=9, color=INK2)
    ax.set_yticks(range(len(show))); ax.set_yticklabels(show, fontsize=9, color=INK2)
    ax.set_xlabel("domain b (whose geometry is measured)", color=INK2); ax.set_ylabel("domain a (whose subspace is removed)", color=INK2)
    for i in range(len(show)):
        for j in range(len(show)):
            if np.isfinite(M[i, j]) and i != j and M[i, j] > 0.15:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("share of b's geometry inside a's subspace", color=INK2)
    ax.set_title(f"Which concept manifolds share a subspace? (OLMo-2-7B at {tag} tokens, {tmpl} prompts)", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(f"{out}_overlap.png", facecolor=SURFACE)


def loo_pred(H, T, alpha=1.0):
    n = len(H)
    Hc = H - H.mean(0)
    ev, U = np.linalg.eigh(Hc @ Hc.T)
    S = (U * (ev / (ev + alpha * ev.sum() / n))) @ U.T + 1.0 / n
    return T - (T - S @ T) / (1 - np.diag(S))[:, None]


def gallery(res, ddir, out, tag, tmpl):
    D = domains()
    z = np.load(Path(ddir) / f"states_{tag}.npz")
    S, owner, layers = z[tmpl].astype(np.float64), z["owner"], list(z["layers"])
    show = order(res[tag])
    nc = 6; nr = int(np.ceil(len(show) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.2 * nr), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    for ax, k in zip(axs.flat, show):
        d = D[k]; prim = res[tag]["domains"][k]["primary"]
        fits = res[tag][tmpl]["fits"][k][prim]
        # the saved layer (8 / 16 / 24) where this domain's geometry is best decoded
        li = max(range(len(layers)), key=lambda i: fits[res[tag]["layers"].index(layers[i])]["r2"])
        H = S[li, owner == k]
        kind = d["kind"]
        if kind in ("cyclic",):
            T = d["targets"]["circle" if "circle" in d["targets"] else ("chromatic" if "chromatic" in d["targets"] else "hue")]
            P = loo_pred(H, T[:, :2])
            c = d["coord"]
        elif kind in ("helix", "linear", "magnitude", "kin", "scalar"):
            T = d["targets"]["line" if "line" in d["targets"] else prim]
            if kind == "scalar":   # separate scales: the within-scale position, coloured by scale
                T = d["targets"]["within"].sum(1, keepdims=True)
            P = np.stack([T[:, 0], loo_pred(H, T[:, :1])[:, 0]], 1)
            c = np.unique(d["group"], return_inverse=True)[1] if kind == "scalar" else np.asarray(d["coord"], dtype=float)
        elif kind == "tree":   # predicted hierarchy code, first two principal axes, coloured by top-level category
            Pt = loo_pred(H, d["targets"]["tree"])
            Pt = Pt - Pt.mean(0)
            P = Pt @ np.linalg.svd(Pt, full_matrices=False)[2][:2].T
            c = np.unique(d["label"], return_inverse=True)[1]
        elif kind == "sphere":
            T = d["targets"]["plane"]   # (lat, lon) standardised; show predicted vs true map
            P = loo_pred(H, T)[:, ::-1]
            c = d["coord"][:, 1]
        else:
            T = d["targets"]["plane"]
            P = loo_pred(H, T)
            if k == "us_cities":   # (lat, lon) -> map orientation
                P = P[:, ::-1]
            c = d["coord"][:, 0]
        ax.scatter(P[:, 0], P[:, 1], c=c, cmap="twilight" if kind == "cyclic" else "viridis", s=16, edgecolors="none")
        if kind in ("cyclic",):
            o = np.argsort(c); ax.plot(P[o, 0], P[o, 1], color=GRID, linewidth=0.8, zorder=0)
        if kind in ("helix", "linear", "magnitude", "kin", "scalar"):
            ax.plot([-2, 2], [-2, 2], color=GRID, linewidth=0.8, zorder=0)
        r2 = fits[res[tag]["layers"].index(layers[li])]
        ax.set_title(f"{k}  (L{layers[li]}, R² {r2['r2']:.2f})", fontsize=9.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
    for ax in list(axs.flat)[len(show):]:
        ax.axis("off")
    fig.suptitle(f"Concept manifolds at {tag}: leave-one-out decoded coordinates "
                 "(circles: predicted cos/sin; lines / magnitudes / scales: true vs predicted position; places: predicted "
                 "longitude × latitude; trees: predicted hierarchy code, 2 PCs; emotions: valence × arousal)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{out}_gallery.png", facecolor=SURFACE)


def families(res, out, tags, tmpl):
    """Domain x model: geometry score (best-layer R^2 - best-layer null) and rescaled test accuracy."""
    names = order(res[tags[0]])
    G = np.array([[geo_score(res[t], tmpl, k, res[t]["domains"][k]["primary"]) for t in tags] for k in names])
    A = np.array([[(res[t]["tasks"][k]["acc"] - res[t]["tasks"][k]["chance"]) / (1 - res[t]["tasks"][k]["chance"])
                   for t in tags] for k in names])
    fig, axs = plt.subplots(1, 2, figsize=(4 + 1.3 * len(tags) * 2, 0.32 * len(names) + 2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, M, t in zip(axs, (G, A), ("geometry (R² − null, best layer)", "3-shot test (chance 0, perfect 1)")):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=9, color=INK2)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9, color=INK2)
        for i in range(len(names)):
            for j in range(len(tags)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] < 0.5 else INK)
        ax.set_title(t, color=INK, fontsize=11.5, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"Which concept manifolds are universal across model families? ({tmpl} prompts)", color=INK, fontsize=14,
                 x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}_families.png", facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="checkpoint for the overlap and gallery figures (default: the latest)")
    ap.add_argument("--tmpl", default="cued", choices=("cued", "neutral"))
    ap.add_argument("--prefix", default=None, help="atlas v2: one trajectory, e.g. olmo2-7b-")
    ap.add_argument("--families", default=None, help="atlas v2: comma-separated final-checkpoint tags for the families figure")
    a = ap.parse_args()
    if a.families:
        tags = a.families.split(",")
        families(load(a.dir), a.out, tags, a.tmpl)
        print("wrote", f"{a.out}_families.png", tags)
        return
    res = load(a.dir, a.prefix)
    tag = a.tag or list(res)[-1]
    timeline(res, a.out, a.tmpl)
    overlap(res, a.out, tag, a.tmpl)
    gallery(res, a.dir, a.out, tag, a.tmpl)
    print("wrote", a.out, "checkpoints", list(res), "detail at", tag)


if __name__ == "__main__":
    main()
