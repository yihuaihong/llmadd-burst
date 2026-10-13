"""Summary of from-scratch pretraining runs (tools/pretrain_helix.py), pooled by condition over seeds.

Condition = arm + window (helix_window) / gate threshold (helix_gate) + --geo_site + --split + arithmetic lines; runs
from several directories are pooled by condition (diagnostic reruns with tags starting "_rerun" / "_i" are skipped).
Metric key (--key): default = the held-out set in training position of the run's split (uheld_ctx for
--split unordered, test_ctx for the original ordered split; NOT the position-0 "test", which the prior depresses).
Per condition, mean over seeds of:
  final; peak; area (mean over the curve);
  t50 / t90 in millions of tokens, two definitions printed side by side:
    first  = the first eval at which the key reaches the threshold
    stays  = the first eval after which it never drops below the threshold again
  (median over seeds; "-" if not reached by some seed, with the count reached).
Tests vs the "none" condition with the same split and arithmetic share: Welch t-test on area and on final.

    python tools/pt_summary.py <run dirs...> [--key uheld_ctx]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json

import numpy as np
from scipy import stats


def label(a: dict) -> str:
    s = a["arm"]
    if a["arm"] in ("helix_window", "helix_shuf_window"):
        s += f" {a.get('window')}" + (f" after={a['lam_after']}" if a.get("lam_after") else "")
    if a["arm"] == "helix_gate":
        s += f" thr={a.get('gate_thr')} start={a.get('gate_start')}"
    if a["arm"] in ("helix_acc", "helix_cka"):
        s += f" thr={a.get('gate_thr')}" if a["arm"] == "helix_acc" else f" rel={a.get('cka_release')}"
    if a.get("geo_site", "prefix") != "prefix" and a["arm"] != "none":
        s += f" @{a['geo_site']}"
    return s


def t_reach(curve, key, thr, stays):
    toks = [e["tokens"] / 1e6 for e in curve]
    v = [e[key] for e in curve]
    if not stays:
        return next((t for t, x in zip(toks, v) if x >= thr), np.nan)
    for i in range(len(v)):
        if all(x >= thr for x in v[i:]):
            return toks[i]
    return np.nan


def fmt_t(vals):
    vals = np.array(vals, dtype=float)
    ok = np.isfinite(vals)
    if ok.all():
        return f"{np.median(vals):.0f}"
    return f"- ({ok.sum()}/{len(vals)})" if not ok.any() else f"{np.median(vals[ok]):.0f} ({ok.sum()}/{len(vals)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    runs = collections.defaultdict(list)
    for d in args.dirs:
        for p in sorted(set(glob.glob(f"{d}/run_*.json") + glob.glob(f"{d}/out/run_*.json") + glob.glob(f"{d}/out/*/run_*.json"))):
            r = json.load(open(p))
            a = r["args"]
            if str(a.get("tag") or "").startswith(("_rerun", "_i")):
                continue
            runs[(a.get("split", "ordered"), a.get("arith_lines") or 1, label(a))].append(r)
    # the same condition and seed in two batches (e.g. pt7c and ptfinal) is the same init and data stream: keep one
    for k, rs in runs.items():
        by_seed = {}
        for r in rs:
            sd = r["args"]["seed"]
            if sd not in by_seed or len(r["curve"][-1]) > len(by_seed[sd]["curve"][-1]):
                by_seed[sd] = r
        if len(by_seed) < len(rs):
            print(f"note: {k[2]}: {len(rs) - len(by_seed)} duplicate seed run(s) dropped (kept the one with most eval keys)")
        runs[k] = list(by_seed.values())
    groups = sorted(runs, key=lambda k: (k[0], k[1], k[2] != "none", k[2]))
    for (split, lines, lab) in groups:
        rs = runs[(split, lines, lab)]
        key = args.key or ("uheld_ctx" if split == "unordered" else "test_ctx")
        rs = [r for r in rs if key in r["curve"][-1]]
        if not rs:
            continue
        fin = [100 * r["curve"][-1][key] for r in rs]
        area = [100 * np.mean([e[key] for e in r["curve"]]) for r in rs]
        base = [r for r in runs.get((split, lines, "none"), []) if key in r["curve"][-1]]
        test = ""
        if base and lab != "none":
            ba = [100 * np.mean([e[key] for e in r["curve"]]) for r in base]
            bf = [100 * r["curve"][-1][key] for r in base]
            test = (f" | vs none: area p {stats.ttest_ind(area, ba, equal_var=False).pvalue:.3f}, "
                    f"final p {stats.ttest_ind(fin, bf, equal_var=False).pvalue:.3f}")
        ts = {f"t{int(100 * th)} {'stays' if st else 'first'}": fmt_t([t_reach(r['curve'], key, th, st) for r in rs])
              for th in (0.5, 0.9) for st in (False, True)}
        print(f"[{split}, {lines} line(s), key {key}] {lab:40s} n={len(rs)}  final {np.mean(fin):5.1f}  "
              f"area {np.mean(area):5.1f}  " + "  ".join(f"{k} {v}" for k, v in ts.items()) + test)


if __name__ == "__main__":
    main()
