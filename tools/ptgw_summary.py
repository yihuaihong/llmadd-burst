"""Summary and pre-registered decision for runs/ptgw: window (W) vs delayed gate (D05: thr 0.5; D09h: thr 0.9 +
hysteresis 0.1) across settings (main, lr5e-4, Lall, tok6e8). Implements runs/ptgw/AUDIT.md including the rule
amendment of 2026-10-15 (written before any result): validity checks (exit 1 unless --allow_invalid), symmetric
"worse" between all schedules, paired worst-case regret, the decision table, descriptive non-inferiority.

Per run: b33_area (mean over evals of held_b33 x 100, PRIMARY speed), area6, b33_last5, b33_t90 (first eval after
which held_b33 >= 0.9 for good; steps and LR-weighted progress), bpb (logged, 65k tokens), bpb_offline (PRIMARY LM
cost; tools/offline_lm.py, used only if its double path passed and it covers all of val.bin), prior_on_frac, toggles
(excluding the onset switch for delayed schedules), first close, held_b33 at the first close.

    python tools/ptgw_summary.py [--root /scratch/yh6210/llmadd_torch/runs] [--offline <offline_lm json> ...]
                                 [--allow_invalid]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

B = ("b11", "b12", "b13", "b22", "b23", "b33")
SETTINGS = ("main", "lr5e-4", "Lall", "tok6e8")
SCHED = ("W", "D05", "D09h")
# expected args per cell (mirror of llmadd_torch/make_jobs_ptgw.py)
EXP_SET = {"main": dict(lr=1e-3, geo_layers=None, tokens=3e8), "lr5e-4": dict(lr=5e-4, geo_layers=None, tokens=3e8),
           "Lall": dict(lr=1e-3, geo_layers="1,2,3,4,5,6", tokens=3e8), "tok6e8": dict(lr=1e-3, geo_layers=None, tokens=6e8)}
EXP_SCHED = {"W": dict(target="helix3", schedule="window", window="0.05,0.4"),
             "D05": dict(target="helix3", schedule="gate", gate_start=0.05, gate_thr=0.5, gate_hyst=0.0),
             "D09h": dict(target="helix3", schedule="gate", gate_start=0.05, gate_thr=0.9, gate_hyst=0.1),
             "none": dict(target="none")}
MARGIN = {"b33_area": 2.0, "area6": 1.0}
BPB_MARGIN = 0.004
HIGHER_BETTER = {"b33_area": 1, "area6": 1, "b33_last5": 1, "bpb": -1, "bpb_offline": -1, "b33_t90_lrw": -1}
REFS = {("main", "none"): "pt999/out/7pct-none", ("main", "G0"): "pt999/out/7pct-gate-ctx",
        ("main", "whole"): "pt999/out/7pct-whole-ctx", ("main", "W_pt999b"): "pt999b/out/7pct-win-ctx",
        ("lr5e-4", "none_pt999"): "pt999/out/sens-lr5e-4-none", ("lr5e-4", "G0"): "pt999/out/sens-lr5e-4-gate-ctx",
        ("Lall", "G0"): "pt999/out/sens-Lall-gate-ctx"}
PROBLEMS = []


def lr_factor(s, steps):
    warm = max(1, steps // 100)
    return min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))


def metrics(r, f, off, ptgw):
    c, steps, a = r["curve"], r["steps"], r["args"]
    b33 = [e["held_b33"] for e in c]
    t90 = next((e["step"] for i, e in enumerate(c) if all(v >= 0.9 for v in b33[i:])), None)
    cum = np.cumsum([lr_factor(s, steps) for s in range(steps)])
    ev = r.get("prior_events", [])
    delayed = a.get("schedule") == "window" or (a.get("schedule") == "gate" and a.get("gate_start", 0) > 0)
    first_close = next((s for s, on in ev if on == 0), None)
    out = {"b33_area": 100 * np.mean(b33), "area6": 100 * np.mean([np.mean([e[f"held_{b}"] for b in B]) for e in c]),
           "b33_last5": 100 * np.mean(b33[-5:]), "b33_t90_steps": t90,
           "b33_t90_lrw": None if t90 is None else float(cum[t90 - 1] / cum[-1]) if t90 > 0 else 0.0,
           "bpb": c[-1]["val_bits_per_byte"], "prior_on_frac": c[-1].get("prior_on_frac"),
           "toggles": None if c[-1].get("prior_toggles") is None else c[-1]["prior_toggles"] - int(delayed and bool(ev)),
           "first_close": first_close, "never_closed": a.get("schedule") == "gate" and first_close is None,
           "b33_at_close": None if first_close is None else float(np.interp(first_close, [e["step"] for e in c], b33)),
           "code_sha": r.get("code_sha"), "steps": steps, "curve": c, "events": ev,
           "b33_area_first3e8": 100 * np.mean([e["held_b33"] for e in c if e["step"] < 18310])}
    out["bpb_offline"] = None
    if off is not None:
        bad = []
        if abs(off.get("orig_minus_logged", 1.0)) >= 0.002:
            bad.append("double path")
        if off.get("tokens_full", 0) < 1_900_000 or off.get("max_windows", 0):
            bad.append("partial val")
        if Path(off.get("run", "")).resolve() != Path(f).resolve():
            bad.append("other run file")
        if off.get("bpb_logged_final") != c[-1]["val_bits_per_byte"]:
            bad.append("stale (run rewritten)")
        if bad:
            (PROBLEMS if ptgw else []).append(f"offline entry not used for {f}: {bad}")
        else:
            out["bpb_offline"] = off["bpb_full"]
    return out


def check_args(r, st, sc, f):
    a = r["args"]
    exp = dict(EXP_SCHED[sc], site="ctx", arith_lines=4, **EXP_SET[st])
    for k, v in exp.items():
        got = a.get(k)
        same = (got == v) if not isinstance(v, float) else (got is not None and abs(float(got) - v) < 1e-12 * max(1, abs(v)))
        if not same:
            PROBLEMS.append(f"args mismatch {f}: {k}={got!r}, expected {v!r}")
    if sc in ("D05", "D09h"):
        first = math.ceil(0.05 * r["steps"])
        ev = r.get("prior_events", [])
        if not ev or ev[0] != [first, 1]:
            PROBLEMS.append(f"delayed gate did not switch on at step {first}: {f} events {ev[:3]}")


def load(d, offline, cell=None):
    runs = {}
    for f in sorted(glob.glob(f"{d}/run_*.json")):
        r = json.loads(Path(f).read_text())
        seed = r["args"]["seed"]
        if r["curve"][-1]["step"] < r["steps"] - 1:
            if cell:
                PROBLEMS.append(f"unfinished: {f}")
            continue
        if seed in runs:
            PROBLEMS.append(f"two runs for seed {seed} in {d}")
            continue
        if cell:
            check_args(r, *cell, f)
        runs[seed] = metrics(r, f, offline.get(str(Path(f).resolve())), bool(cell))
    if cell and sorted(runs) != [0, 1, 2, 3]:
        PROBLEMS.append(f"cell {cell} has seeds {sorted(runs)} (needs 0-3)")
    return runs


def signflip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    return float(np.mean([abs(np.mean(d * np.array(sg))) >= obs - 1e-12 for sg in itertools.product((1, -1), repeat=len(d))]))


def paired(x, y):
    dd = np.asarray(x, float) - np.asarray(y, float)
    n = len(dd)
    sd = dd.std(ddof=1) if n > 1 else float("nan")
    p = stats.ttest_rel(x, y).pvalue if n > 1 and sd > 0 else (0.0 if n > 1 and dd.mean() != 0 else float("nan"))
    half = stats.t.ppf(0.95, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return dd, sd, p, half


def worse(cells, st, S, T, key):
    """S worse than T in setting st on key (symmetric rule of the 2026-10-15 amendment)."""
    s_, t_ = cells[(st, S)], cells[(st, T)]
    common = sorted(set(s_) & set(t_))
    x, y = [s_[i][key] for i in common], [t_[i][key] for i in common]
    if len(common) < 2 or any(v is None for v in x + y):
        return False, None
    dd, sd, p, _ = paired(x, y)
    need = math.ceil(0.75 * len(dd))
    if key == "bpb_offline":   # higher = worse
        hit = (dd > BPB_MARGIN).sum() >= need and dd.mean() > BPB_MARGIN and p < .05
    else:
        hit = (dd < -MARGIN[key]).sum() >= need and dd.mean() < 0 and p < .05
    return bool(hit), (dd, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/yh6210/llmadd_torch/runs")
    ap.add_argument("--offline", nargs="*", default=[])
    ap.add_argument("--allow_invalid", action="store_true")
    a = ap.parse_args()
    offline = {}
    for f in a.offline:
        offline.update(json.loads(Path(f).read_text()))
    root = Path(a.root)
    cells = {}
    for st in SETTINGS:
        for sc in SCHED + (("none",) if st in ("lr5e-4", "tok6e8") else ()):
            cells[(st, sc)] = load(root / "ptgw" / "out" / f"{st}-{sc}", offline, cell=(st, sc))
    shas = {x["code_sha"] for k, rs in cells.items() for x in rs.values()}
    snap = root / "ptgw" / "code" / "tools" / "pretrain_range.py"
    want = hashlib.sha256(snap.read_bytes()).hexdigest()[:12] if snap.exists() else None
    if len(shas) != 1 or (want and shas != {want}):
        PROBLEMS.append(f"code_sha across ptgw: {sorted(map(str, shas))}, snapshot {want}")
    for k, p in REFS.items():
        cells[k] = load(root / p, offline)
    have_offline = all(x["bpb_offline"] is not None for st in SETTINGS for sc in SCHED for x in cells[(st, sc)].values())
    if PROBLEMS:
        print("INVALID (AUDIT validity):\n  " + "\n  ".join(PROBLEMS))
        if not a.allow_invalid:
            sys.exit(1)
        print("--allow_invalid: everything below is DESCRIPTIVE ONLY\n")

    keys = ("b33_area", "area6", "b33_last5", "b33_t90_steps", "b33_t90_lrw", "bpb", "bpb_offline", "prior_on_frac",
            "toggles", "first_close", "b33_at_close")
    print("== cells: mean over seeds [values available / n]")
    for (st, sc), rs in cells.items():
        if not rs:
            continue
        row = []
        for k in keys:
            v = [x[k] for x in rs.values() if x.get(k) is not None]
            row.append(f"{k} {np.mean(v):.4g}" + ("" if len(v) == len(rs) else f" [{len(v)}/{len(rs)}]") if v else f"{k} -")
        never = sum(x["never_closed"] for x in rs.values())
        print(f"  {st:7s} {sc:9s} n={len(rs)} | " + " | ".join(row) + (f" | gate never closed in {never}" if never else ""))
        for s, x in rs.items():
            if sc == "D09h" and x["toggles"] is not None and x["toggles"] > 20:
                print(f"    FLAG D09h {st} seed {s}: {x['toggles']} toggles (> 20: the gate failed in this run)")

    print("\n== paired D - W within setting (per-seed differences, mean, 90% t-CI, paired SD, t p, sign-flip p, D better)")
    for st in SETTINGS:
        for sc in ("D05", "D09h"):
            d_, w_ = cells[(st, sc)], cells[(st, "W")]
            common = sorted(set(d_) & set(w_))
            if len(common) < 2:
                continue
            print(f"  {st} {sc}-W (seeds {common}):")
            for k in ("b33_area", "area6", "b33_last5", "bpb", "bpb_offline", "b33_t90_lrw"):
                x, y = [d_[s].get(k) for s in common], [w_[s].get(k) for s in common]
                if any(v is None for v in x + y):
                    print(f"     {k:12s} n/a")
                    continue
                dd, sd, p, half = paired(x, y)
                better = int((HIGHER_BETTER[k] * dd > 0).sum())
                print(f"     {k:12s} {dd.mean():+.4g} [{dd.mean() - half:+.4g}, {dd.mean() + half:+.4g}] sd {sd:.3g} "
                      f"t p {p:.3f} sf p {signflip_p(dd):.3f} D better {better}/{len(dd)} | dd {np.round(dd, 4).tolist()}")
            dd, sd, p, half = paired([d_[s]["b33_area"] for s in common], [w_[s]["b33_area"] for s in common])
            lo = dd.mean() - half
            bp = [d_[s]["bpb_offline"] for s in common] + [w_[s]["bpb_offline"] for s in common]
            bpb_ok = None
            if None not in bp:
                db, _, _, hb = paired(bp[:len(common)], bp[len(common):])
                bpb_ok = db.mean() + hb < BPB_MARGIN
            tag = ("NON-INFERIOR" if lo > -2 and bpb_ok else
                   "undecided (offline LM missing)" if lo > -2 and bpb_ok is None else "undecided")
            print(f"     descriptive non-inferiority: b33 CI low {lo:+.2f} (> -2?), bpb CI ok {bpb_ok} -> {tag}"
                  f"{'' if sd <= 1.5 else f' (paired SD {sd:.2f} > 1.5: no discriminative power)'}")
            dc = [abs(e1["held_b33"] - e2["held_b33"]) for s in common for e1, e2 in zip(d_[s]["curve"], w_[s]["curve"])
                  if d_[s]["first_close"] is not None and e1["step"] < d_[s]["first_close"]]
            if dc:
                print(f"     before D's first close: max |D - W| held_b33 = {max(dc):.3f} (GPU nondeterminism floor)")
            if st == "tok6e8":
                d3, _, p3, h3 = paired([d_[s]["b33_area_first3e8"] for s in common], [w_[s]["b33_area_first3e8"] for s in common])
                print(f"     first 3e8 tokens only: b33_area {d3.mean():+.3g} ± {h3:.3g} (t p {p3:.3f})")

    print("\n== symmetric 'worse' (S worse than T in a setting)")
    flags = {sc: [] for sc in SCHED}
    for st in SETTINGS:
        for S in SCHED:
            for T in SCHED:
                if S == T:
                    continue
                for key in ("b33_area", "area6") + (("bpb_offline",) if have_offline else ()):
                    hit, info = worse(cells, st, S, T, key)
                    if hit:
                        flags[S].append(f"{st}: {S} worse than {T} on {key} (mean {info[0].mean():+.4g}, p {info[1]:.3f})")
    for sc in SCHED:
        print(f"  {sc:5s} " + ("; ".join(flags[sc]) if flags[sc] else "never worse"))

    def regrets(unit):
        out = {}
        for st in SETTINGS:
            common = sorted(set.intersection(*(set(cells[(st, s)]) for s in SCHED)))
            if not common:
                return None
            means = {s: np.mean([cells[(st, s)][i]["b33_area"] for i in common]) for s in SCHED}
            scale = 2.0 if (unit == "tokens" and st == "tok6e8") else 1.0
            for s in SCHED:
                out.setdefault(s, []).append((st, scale * (max(means.values()) - means[s])))
        return {s: (max(v for _, v in r), r) for s, r in out.items()}

    def decide(reg):
        """(outcome code, text): code is a schedule name, 'W(tie)', 'user(tie:...)' or 'none'."""
        eligible = [s for s in SCHED if not flags[s]]
        if not eligible:
            return "none", "NO WINNER: every schedule is worse somewhere -> per-setting trade-offs to the user"
        ranked = sorted(eligible, key=lambda s: reg[s][0])
        best = ranked[0]
        if all(reg[s][0] - reg[best][0] > 2 for s in ranked[1:]):
            return best, f"{best} wins (worst-case regret {reg[best][0]:.2f}; eligible {eligible})"
        tied = [s for s in ranked if reg[s][0] - reg[best][0] <= 2]
        if "W" in tied:
            return "W(tie)", f"TIE within 2 points among {tied} -> W by default (report to the user)"
        return f"user(tie:{','.join(sorted(tied))})", f"TIE between {tied} (W not eligible or not tied) -> the user chooses"

    print("\n== worst-case regret on b33_area (paired on common seeds); units: fraction of training / token-equivalent")
    rf, rt = regrets("fraction"), regrets("tokens")
    if rf is None:
        print("  regret undefined: a setting has no seed common to all three schedules")
        return
    for s in SCHED:
        print(f"  {s:5s} fraction: " + " ".join(f"{st}:{v:.2f}" for st, v in rf[s][1]) + f" | worst {rf[s][0]:.2f}   "
              f"tokens: worst {rt[s][0]:.2f}")
    (c1, d1), (c2, d2) = decide(rf), decide(rt)
    lm = "" if have_offline else " [LM part PROVISIONAL: offline bpb missing or invalid for some runs]"
    if c1 != c2:
        print(f"\nDECISION: UNIT-DEPENDENT -> to the user (fraction units: {d1} | token units: {d2}){lm}")
    else:
        print(f"\nDECISION: {d1}{lm}")


if __name__ == "__main__":
    main()
