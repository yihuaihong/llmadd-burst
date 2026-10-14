"""Summary tables for the atlas v2 run (tools/manifold_atlas.py outputs atlas_<model>-<tokens>.json).

geometry = best-layer R^2 of the primary target minus the best-layer permutation null (cued prompts; pooled R^2 for
tree kinds), flagged '*' when not significant (best-layer p >= 0.05); ability = calibrated 3-shot accuracy acc_cal
rescaled (acc - chance) / (1 - chance); prior = the candidate prior alone (acc_prior, rescaled), i.e. how far the test
can be passed without the question. Prints: (1) random-init controls, (2) per trajectory, the first checkpoint where
geometry / ability reach half of their final value, (3) families: final checkpoints side by side.

    python tools/atlas2_summary.py --dir <atlas2 out dir> [--tmpl cued]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

TRAJ = {"olmo2-7b": ["1B", "4B", "9B", "21B", "42B", "84B", "210B", "630B", "3896B", "main"],
        "olmo2-1b": ["1B", "21B", "42B", "84B", "210B", "630B", "4001B", "main"],
        "pythia-6.9b": ["4B", "21B", "84B", "210B", "300B"]}
FINALS = ["olmo2-7b-main", "olmo2-1b-main", "olmo3-7b", "pythia-6.9b-300B", "smollm2-1.7b", "qwen2.5-7b", "qwen3-8b-base",
          "mistral-7b"]


def geo(r, k, tmpl):
    d = r["domains"][k]
    b = r[tmpl]["best"][k][d["primary"]]
    if d.get("pooled"):
        return b["r2p"] - b["null_p"], b["p_p"]
    return b["r2"] - b["null"], b["p"]


def ability(t, key="acc_cal"):
    return (t[key] - t["chance"]) / (1 - t["chance"])


def half_point(tags, vals):
    final = vals[-1]
    if not np.isfinite(final) or final <= 0.05:
        return "-"
    for t, v in zip(tags, vals):
        if v >= 0.5 * final:
            return t
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tmpl", default="cued")
    a = ap.parse_args()
    R = {p.stem.removeprefix("atlas_"): json.loads(p.read_text()) for p in Path(a.dir).glob("atlas_*.json")}
    names = list(next(iter(R.values()))["domains"])

    print("== (1) random-init controls: domains whose geometry beats the null (p < 0.05) without training")
    for tag in ("olmo2-7b-rand", "olmo2-1b-0B"):
        if tag in R:
            sig = [(k, round(geo(R[tag], k, a.tmpl)[0], 2)) for k in names if geo(R[tag], k, a.tmpl)[1] < 0.05]
            print(f"  {tag}: {len(sig)}/{len(names)} significant: {sig}")
            ab = {k: round(ability(R[tag]['tasks'][k]), 2) for k in names}
            big = {k: v for k, v in ab.items() if abs(v) >= 0.3}
            print(f"    calibrated ability far from chance (|x| >= 0.3; n is small for some tests): {big}")

    for fam, cks in TRAJ.items():
        tags = [f"{fam}-{c}" for c in cks if f"{fam}-{c}" in R]
        if len(tags) < 3:
            continue
        print(f"\n== (2) {fam}: {len(tags)} checkpoints {[t.rsplit('-', 1)[1] for t in tags]}")
        print(f"  {'domain':15s} {'geo final':>9s} {'geo half':>9s} {'abil final':>10s} {'abil half':>9s}  geometry per checkpoint")
        for k in names:
            g = [geo(R[t], k, a.tmpl)[0] for t in tags]
            ab = [ability(R[t]["tasks"][k]) for t in tags]
            sig_final = geo(R[tags[-1]], k, a.tmpl)[1] < 0.05
            print(f"  {k:15s} {g[-1]:9.2f}{'' if sig_final else '*'} {half_point([t.rsplit('-', 1)[1] for t in tags], g):>9s} "
                  f"{ab[-1]:10.2f} {half_point([t.rsplit('-', 1)[1] for t in tags], ab):>9s}  "
                  + " ".join(f"{v:5.2f}" for v in g))

    fin = [t for t in FINALS if t in R]
    print(f"\n== (3) families, final checkpoints: geometry (ability) ; '*' = geometry not significant")
    print(f"  {'domain':15s} " + " ".join(f"{t[:13]:>14s}" for t in fin))
    for k in names:
        cells = []
        for t in fin:
            g, p = geo(R[t], k, a.tmpl)
            cells.append(f"{g:5.2f}{'*' if p >= 0.05 else ' '}({ability(R[t]['tasks'][k]):4.1f})")
        print(f"  {k:15s} " + " ".join(f"{c:>14s}" for c in cells))
    univ = [k for k in names if all(geo(R[t], k, a.tmpl)[1] < 0.05 and geo(R[t], k, a.tmpl)[0] >= 0.3 for t in fin)]
    print(f"\n  geometry >= 0.3 and significant in ALL {len(fin)} final models: {univ}")


if __name__ == "__main__":
    main()
