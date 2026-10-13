"""Summary of from-scratch pretraining runs (tools/pretrain_helix.py), pooled by condition over seeds.

Whether a run learns two-digit addition is chaotic in the 2%-arithmetic setting (F61: identical reruns end at 0-53%),
so each condition is summarised by its distribution over seeds rather than by single runs:
  final  = held-out accuracy at the end;  peak = best held-out accuracy on the curve;
  area   = mean held-out accuracy over the whole curve;  hit = runs whose peak reaches --hit (default 20%);
  t50    = tokens (M) at which the mean curve first reaches half of its own peak, when that is defined.
Tests against the "none" condition: Mann-Whitney on area (one-sided, greater) and Fisher exact on hits.

    python tools/pt_summary.py /scratch/yh6210/llmadd_torch/runs/pt-* [--hit 0.2]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json

import numpy as np
from scipy import stats


def label(a: dict) -> str:
    if a["arm"] in ("none", "helix", "helix_shuf"):
        return a["arm"]
    s = a["arm"]
    if a["arm"] == "helix_window":
        s += f" {a.get('window')}"
    for k in ("lam_after", "gate_thr", "cka_release", "arith_lines"):
        if a.get(k) not in (None, 0, 0.0, "", False):
            s += f" {k}={a[k]}"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--hit", type=float, default=0.2)
    args = ap.parse_args()
    runs = collections.defaultdict(list)
    for d in args.dirs:
        for p in glob.glob(f"{d}/out/run_*.json") + glob.glob(f"{d}/run_*.json"):
            r = json.load(open(p))
            if r["args"].get("tag"):      # diagnostic reruns (pt-diag) are not seeds of a condition
                continue
            runs[(label(r["args"]), r["args"].get("arith_lines"))].append(r)
    base = next((v for (lab, _), v in runs.items() if lab == "none"), None)
    area = lambda r: float(np.mean([c["test"] for c in r["curve"]]))
    peak = lambda r: max(c["test"] for c in r["curve"])
    print(f"| condition | n | final (mean) | peak (mean) | area (mean) | peak >= {100 * args.hit:.0f}% | area vs none (MW p) | hits vs none (Fisher p) |")
    print("|---|---|---|---|---|---|---|---|")
    for (lab, _), rs in sorted(runs.items(), key=lambda kv: (kv[0][0] != "none", kv[0][0])):
        fin = [100 * r["curve"][-1]["test"] for r in rs]
        pk = [100 * peak(r) for r in rs]
        ar = [100 * area(r) for r in rs]
        hits = sum(p >= 100 * args.hit for p in pk)
        if base is not None and lab != "none":
            mw = stats.mannwhitneyu(ar, [100 * area(r) for r in base], alternative="greater").pvalue
            bh = sum(100 * peak(r) >= 100 * args.hit for r in base)
            fp = stats.fisher_exact([[hits, len(rs) - hits], [bh, len(base) - bh]], alternative="greater")[1]
            tests = f"{mw:.3f} | {fp:.3f}"
        else:
            tests = "– | –"
        print(f"| {lab} | {len(rs)} | {np.mean(fin):.1f} | {np.mean(pk):.1f} | {np.mean(ar):.1f} | {hits}/{len(rs)} | {tests} |")


if __name__ == "__main__":
    main()
