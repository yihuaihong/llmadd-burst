"""Summary of runs/ptgw: window (W) vs delayed gate (D05: thr 0.5; D09h: thr 0.9 + hysteresis 0.1) across settings
(main, lr5e-4, Lall, tok6e8), with the pre-registered rule of runs/ptgw/AUDIT.md. Per run:
  b33_area   mean over evals of held-out 3+3-digit exact match (x100)      [primary speed metric]
  b33_t90    first eval after which held_b33 >= 0.9 at every later eval; in steps, tokens (M) and LR-weighted
             progress (sum of the LR schedule factor up to that step, so slower-LR phases count less)
  area6      mean over evals of the 6-bucket mean (x100); b33_last5 = mean of the last 5 evals of held_b33 (x100)
  bpb        final logged val bits/byte (65k tokens) and, with --offline <offline_lm.py json>, the full-val value
  gate       prior_on_frac, toggles, first close step, held_b33 at the first close (linear interpolation)
Paired by seed within a setting (same init, data stream, held-out split): D - W, mean, paired t p, exact sign-flip p
(4 seeds: smallest two-sided p = 0.125), wins. Worst-case regret of each schedule across settings on b33_area.
Free references (other snapshots, ctx paths identical by regression test): pt999 7pct-none / 7pct-gate-ctx (gate from
step 0) / 7pct-whole-ctx / sens-lr5e-4-* / sens-Lall-gate-ctx, pt999b 7pct-win-ctx; shown, not used in the rule.

    python tools/ptgw_summary.py --root /scratch/yh6210/llmadd_torch/runs [--offline <json> ...]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

B = ("b11", "b12", "b13", "b22", "b23", "b33")
SETTINGS = ("main", "lr5e-4", "Lall", "tok6e8")
SCHED = ("W", "D05", "D09h")
REFS = {("main", "none"): "pt999/out/7pct-none", ("main", "G0"): "pt999/out/7pct-gate-ctx",
        ("main", "whole"): "pt999/out/7pct-whole-ctx", ("main", "W_pt999b"): "pt999b/out/7pct-win-ctx",
        ("lr5e-4", "none_pt999"): "pt999/out/sens-lr5e-4-none", ("lr5e-4", "G0"): "pt999/out/sens-lr5e-4-gate-ctx",
        ("Lall", "none"): "pt999/out/7pct-none", ("Lall", "G0"): "pt999/out/sens-Lall-gate-ctx"}


def lr_factor(s, steps):
    warm = max(1, steps // 100)
    return min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))


def metrics(r, offline=None):
    c, steps, a = r["curve"], r["steps"], r["args"]
    b33 = [e["held_b33"] for e in c]
    t90 = None
    for i, e in enumerate(c):
        if all(v >= 0.9 for v in b33[i:]):
            t90 = e["step"]
            break
    lrw = (lambda S: sum(lr_factor(s, steps) for s in range(S)) / sum(lr_factor(s, steps) for s in range(steps)))
    ev = r.get("prior_events", [])
    first_close = next((s for s, on in ev if on == 0), None)
    b33_close = None
    if first_close is not None:
        st = [e["step"] for e in c]
        b33_close = float(np.interp(first_close, st, b33))
    out = {"b33_area": 100 * np.mean(b33), "area6": 100 * np.mean([np.mean([e[f"held_{b}"] for b in B]) for e in c]),
           "b33_last5": 100 * np.mean(b33[-5:]), "b33_final": 100 * b33[-1],
           "b33_t90_steps": t90, "b33_t90_tokensM": None if t90 is None else t90 * a["bs"] * a["seq"] / 1e6,
           "b33_t90_lrw": None if t90 is None else lrw(t90),
           "bpb": c[-1]["val_bits_per_byte"], "prior_on_frac": c[-1].get("prior_on_frac"),
           "toggles": c[-1].get("prior_toggles"), "first_close": first_close, "b33_at_close": b33_close}
    if offline is not None:
        out["bpb_offline"] = offline.get("bpb_full")
    return out


def signflip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    sims = [abs(np.mean(d * np.array(sg))) for sg in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([v >= obs - 1e-12 for v in sims]))


def load(d, offline):
    runs = {}
    for f in sorted(glob.glob(f"{d}/run_*.json")):
        r = json.loads(Path(f).read_text())
        if r["curve"][-1]["step"] < r["steps"] - 1:
            print("unfinished:", f)
            continue
        seed = r["args"]["seed"]
        assert seed not in runs, f"two runs for seed {seed} in {d}"
        off = offline.get(f"{Path(d).name}/seed{seed}")
        runs[seed] = metrics(r, off)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs")
    ap.add_argument("--offline", nargs="*", default=[])
    a = ap.parse_args()
    offline = {}
    for f in a.offline:
        offline.update(json.loads(Path(f).read_text()))
    root = Path(a.root)
    cells = {}
    for st in SETTINGS:
        for sc in SCHED + ("none",):
            cells[(st, sc)] = load(root / "ptgw" / "out" / f"{st}-{sc}", offline)
    for k, p in REFS.items():
        cells[k] = load(root / p, offline)
    keys = ("b33_area", "b33_t90_steps", "b33_t90_lrw", "area6", "b33_last5", "bpb", "bpb_offline", "prior_on_frac",
            "toggles", "first_close", "b33_at_close")
    print("== cells (mean over seeds; n)")
    for (st, sc), rs in cells.items():
        if not rs:
            continue
        row = []
        for k in keys:
            v = [x[k] for x in rs.values() if x.get(k) is not None]
            row.append(f"{k} {np.mean(v):.4g}" if v else f"{k} -")
        print(f"  {st:7s} {sc:9s} n={len(rs)} | " + " | ".join(row))
    print("\n== paired within setting: D - W (mean, paired t p, sign-flip p, wins)")
    for st in SETTINGS:
        for sc in ("D05", "D09h"):
            d_, w_ = cells[(st, sc)], cells[(st, "W")]
            common = sorted(set(d_) & set(w_))
            if len(common) < 2:
                continue
            parts = []
            for k in ("b33_area", "area6", "b33_last5", "bpb", "bpb_offline", "b33_t90_lrw"):
                x = [d_[s].get(k) for s in common]; y = [w_[s].get(k) for s in common]
                if any(v is None for v in x + y):
                    parts.append(f"{k} n/a")
                    continue
                dd = np.array(x) - np.array(y)
                p = stats.ttest_rel(x, y).pvalue if np.std(dd) > 0 else float("nan")
                parts.append(f"{k} {dd.mean():+.4g} (t p {p:.3f}, sf p {signflip_p(dd):.3f}, wins {int((dd > 0).sum())}/{len(dd)})")
            print(f"  {st:7s} {sc:5s}-W n={len(common)}: " + "; ".join(parts))
    print("\n== worst-case regret on b33_area across settings (best schedule mean - this schedule mean)")
    for sc in SCHED:
        regs = []
        for st in SETTINGS:
            means = {s: np.mean([x["b33_area"] for x in cells[(st, s)].values()]) for s in SCHED if cells[(st, s)]}
            if sc in means:
                regs.append((st, max(means.values()) - means[sc]))
        if regs:
            print(f"  {sc:5s} " + " ".join(f"{st}:{v:.2f}" for st, v in regs) + f" | worst {max(v for _, v in regs):.2f}")


if __name__ == "__main__":
    main()
