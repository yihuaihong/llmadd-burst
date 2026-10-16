"""Pre-registered read-out of the pt2main Burst batch (tasks 350-361 and EXTRA 362-399 from tools/make_pt2main_tasks.py;
intent spec and rules report/audits/pt2main_AUDIT.md). Thin wrapper around tools/pt999_summary.py:
  1. collects the returned runs results/<task>/<tag>/run_*.json of every task the generator makes (its tasks(): the 72
     and EXTRA) and prints pt999_summary.py's tables for those task dirs, unchanged (cell lines, per-bucket areas,
     same-lr paired comparisons vs none, win-ctx - win-shuf-ctx). One run per (tag, seed): a re-run (EXTRA, same cell
     and seed, new number) supersedes an unfinished or missing run, or a finished run whose check failed; of several
     finished runs the first (in task order) whose check passed counts, else the first, and the others are reported;
  2. applies the batch's rules, with area / t90 computed by pt999_summary's formulas (its printed cell means are parsed
     and must agree within 0.05, exit 3 otherwise):
     - best lr per (block, arm) = argmax of the mean area over seeds 0-3, over the arm's lr grid (BLOCKS + EXTRA
       cells); every lr within 2 points of the top -> the one with the lower mean bpb, offline (offline_lm.json, full
       val) when every compared run has it, else the in-training bpb for all of them (marked '*'); a diverged run
       keeps its (near-zero) area, so it counts against its lr, and its NaN bpb counts as worst; "open bracket" = the
       best lr is at the edge of the arm's lr grid and beats its neighbour by more than 2 points (an argmax at the
       edge by 2 or less is printed as a note, not an open bracket);
     - "2% speed-up survives tuning" (per block): area(win-ctx@best) - area(none@best) > 0 with >= 3/4 seed wins and
       paired t p < .05 on seeds 0-3, and, when the confirm seeds 4-7 of both cells exist, the same sign with >= 3/4
       wins there; a gap below 10 points without them reads "survival pending: confirm seeds 4-7 needed"; the same
       test for win-text;
     - pt2long headline: t90 ratio none@best / win-ctx@best = ratio of the seed medians of t90 (seeds 0-3), a run that
       never reaches 0.9 counted at the budget (1200M tokens). The median of n values includes a censored one when at
       least (n + 1) // 2 are censored (4 seeds: 2): then the none median is a lower bound and the ratio is written
       '>='; a censored win-ctx median gives no ratio. 90% CI from 10000 paired seed bootstraps (numpy
       default_rng(0)), with the share of resamples whose none median is censored. Printed with the window in tokens
       (the ratio belongs to this budget's schedule, not to pt2b's);
     - control (pt2long, lr 1e-3): win-shuf-ctx must be below win-ctx (mean area difference > 0, >= 3/4 seed wins);
     - b11 separately (natural "a + b = c" with held-out b11 pairs is not excluded from the text): per cell the b11 area
       and the 5-bucket area without b11; the best-vs-best tests are repeated on the latter;
     - integrity: one finished run per (tag, seed); per block and seed the step-0 lm agrees within 1e-3 across cells
       (same init and first batch: the pairing is real); every counted task's check.json passed. A diverged run is
       data, not an integrity failure (its check skips only what a NaN makes meaningless) and is listed.
    python tools/pt2main_summary.py [<results dir>]   (default: results/ of this repo)
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_pt2main_tasks as G   # noqa: E402  (blocks, cells, seeds, EXTRA: one source of truth with the task files)

B = ("b11", "b12", "b13", "b22", "b23", "b33")
MAIN, CONFIRM = tuple(G.SEEDS), tuple(range(4, 8))
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results"


def held(e, bs=B):
    return np.mean([e[f"held_{b}"] for b in bs])


def area(r, bs=B):   # pt999_summary.area over the 6 buckets (or a subset)
    return 100 * np.mean([held(e, bs) for e in r["curve"]])


def area5(r):        # without b11
    return area(r, B[1:])


def t90(r):          # pt999_summary.t90 (M tokens, linear interpolation); nan if 0.9 is never reached
    prev, per_step = None, r["args"]["bs"] * r["args"]["seq"]
    for e in r["curve"]:
        v, tk = held(e), e["step"] * per_step
        if v >= 0.9:
            if prev is None:
                return tk / 1e6
            t0, v0 = prev
            return (t0 + (0.9 - v0) / (v - v0) * (tk - t0)) / 1e6
        prev = (tk, v)
    return np.nan


def budget(r):
    return r["steps"] * r["args"]["bs"] * r["args"]["seq"] / 1e6


def diverged_at(r):  # the check's rule: first eval with a non-finite LM / val loss, prior loss or prior gradient
    fin = lambda v: v is None or math.isfinite(v)
    bad = [e["step"] for e in r["curve"]
           if not (fin(e["lm"]) and fin(e["val_loss"]) and fin(e.get("geo")) and fin(e.get("grad_prior")))]
    return bad[0] if bad else None


def censored_median(x, bud):   # the median of n values includes a censored one (counted at the budget)
    return int((np.asarray(x) >= bud).sum()) >= (len(x) + 1) // 2


# ---------------------------------------------------------------- collect
T = G.tasks()   # [(name, tag, text, args, block, arm, lr, seed)]: the 72 in submission order, then EXTRA
grid = {}       # block -> {arm: [lr, ...]} (BLOCKS cells, then EXTRA lrs)
found = {}      # (tag, seed) -> [(task dir, state, run)] in tasks() order
for name, tag, _, _, block, arm, lr, s in T:
    lrs = grid.setdefault(block, {}).setdefault(arm, [])
    if lr not in lrs:
        lrs.append(lr)
    task = RES / name
    fs = sorted(task.glob(f"{tag}/run_*.json"))
    if not fs:
        found.setdefault((tag, s), []).append((task, "missing", None))
        continue
    assert len(fs) == 1, f"two runs in {task}: {fs}"
    r = json.loads(fs[0].read_text())
    assert r["args"]["tag"] == f"_{tag}" and r["args"]["seed"] == s, (fs[0], r["args"]["tag"], r["args"]["seed"])
    st = "finished" if r["curve"][-1]["step"] >= r["steps"] - 1 else \
        f"unfinished (last eval {r['curve'][-1]['step']} of {r['steps']})"
    found.setdefault((tag, s), []).append((task, st, r))
runs, bpb, problems, dirs, notes = {}, {}, [], [], []
passed = lambda task: (task / "check.json").exists() and json.loads((task / "check.json").read_text())["pass"]
for (tag, s), L in found.items():
    # finished runs, those whose check passed first (task order within each group): a finished run is re-run only when
    # its check failed (e.g. no model after a kill between checkpoint removal and model save)
    fin = sorted([(task, r) for task, st, r in L if st == "finished"], key=lambda x: not passed(x[0]))
    for task, st, _ in L:
        if st != "finished":
            problems.append(f"{st}: {task.name}" + (f" (superseded by {fin[0][0].name})" if fin else ""))
    if not fin:
        continue
    task, r = fin[0]
    if len(fin) > 1:
        problems.append(f"{len(fin)} finished runs of {tag} seed {s}: {task.name} counts (the first whose check passed, "
                        f"else the first), {', '.join(t.name for t, _ in fin[1:])} ignored")
    runs[(tag, s)] = r
    dirs.append(str(task))
    ck = task / "check.json"
    if not passed(task):
        problems.append(f"check.json {'missing' if not ck.exists() else 'FAILED'}: {task.name}")
    off = json.loads((task / "offline_lm.json").read_text()) if (task / "offline_lm.json").exists() else {}
    o = [v for v in off.values() if v.get("tag") == f"_{tag}" and v.get("seed") == s and v.get("max_windows") == 0]
    bpb[(tag, s)] = (o[0]["bpb_full"] if o else None, r["curve"][-1]["val_bits_per_byte"])   # (offline, in-training)
    if diverged_at(r) is not None:
        notes.append(f"diverged (data: its area counts against its lr): {task.name}, first non-finite eval at step "
                     f"{diverged_at(r)}")
for p in problems:
    print("NOTE", p)
for p in notes:
    print("NOTE", p)
if not runs:
    sys.exit(f"no finished pt2main runs under {RES}")

# ---------------------------------------------------------------- 1. pt999_summary.py tables, unchanged
print("=" * 30, "tools/pt999_summary.py on the returned pt2main task dirs (one per tag and seed)")
o = subprocess.run([sys.executable, str(HERE / "pt999_summary.py"), *dirs], capture_output=True, text=True)
print(o.stdout, end="")
if o.returncode:
    print(o.stderr); sys.exit("pt999_summary.py failed")
theirs = {m.group(1): float(m.group(2)) for m in re.finditer(r"^  (\S+)\s+n=\d+ area\s+(-?[\d.]+)±", o.stdout, re.M)}
for tag in {t for t, _ in runs}:   # double path: the same area by both scripts
    mine = np.mean([area(r) for (t, _), r in runs.items() if t == tag])
    if abs(theirs.get(tag, np.inf) - mine) > 0.05:
        sys.exit(f"area of {tag}: pt999_summary {theirs.get(tag)} vs here {mine:.2f}")
print(f"(double path: the mean area of all {len(theirs)} cells agrees with pt999_summary.py within 0.05)")


# ---------------------------------------------------------------- 2. pre-registered rules
def cell(tag, f=area, seeds=MAIN):
    """values over the seeds, or None if any seed is missing / unfinished"""
    v = [f(runs[(tag, s)]) if (tag, s) in runs else None for s in seeds]
    return None if None in v else np.array(v)


def test(t1, t2, f=area, seeds=MAIN):
    """t1 - t2 paired over the seeds: (text, survives, mean difference)"""
    x, y = cell(t1, f, seeds), cell(t2, f, seeds)
    if x is None or y is None:
        return "incomplete", False, None
    d = x - y
    p = stats.ttest_rel(x, y).pvalue if np.std(d) > 0 else float("nan")
    ok = d.mean() > 0 and (d > 0).sum() >= 3 and p < .05
    return f"{d.mean():+6.1f} (p {p:.3f}, wins {int((d > 0).sum())}/{len(d)})", ok, d.mean()


def mean_bpb(tags, seeds=MAIN):
    """{tag: mean bpb}: offline if every run of every tag has it, else in-training for all (second value False);
    a NaN (diverged) counts as worst"""
    off = all(bpb[(t, s)][0] is not None for t in tags for s in seeds)
    v = lambda t, s: bpb[(t, s)][0 if off else 1]
    return {t: np.mean([v(t, s) if math.isfinite(v(t, s)) else np.inf for s in seeds]) for t in tags}, off


print("\n" + "=" * 30, "pt2main pre-registered read-out (report/audits/pt2main_AUDIT.md)")
for base, block, size, data, tokens, plan_h, tlim, _ in G.BLOCKS:
    tags = {(a, lr): f"{block}-lr{lr}-{a}" for a, lrs in grid[block].items() for lr in lrs}
    print(f"\n## {block} ({size}, {tokens} tokens, {data})")
    for s in MAIN + CONFIRM:   # same init and first batch in every cell of a seed
        lm0 = [runs[(t, s)]["curve"][0]["lm"] for t in tags.values() if (t, s) in runs]
        if lm0 and max(lm0) - min(lm0) > 1e-3:
            print(f"  WARNING seed {s}: step-0 lm differs across cells by {max(lm0) - min(lm0):.4f} (pairing broken)")
    print(f"  {'cell':28s} {'area':>12s} {'area5':>6s} {'b11':>6s} {'t90 (M)':>9s} {'bpb':>8s}")
    for (a, lr), t in tags.items():
        ar, a5, b1 = cell(t), cell(t, area5), cell(t, lambda r: area(r, ("b11",)))
        if ar is None:
            print(f"  {t:28s} incomplete ({sum((t, s) in runs for s in MAIN)}/4)")
            continue
        tt = np.array([t90(runs[(t, s)]) for s in MAIN])
        bp, off = mean_bpb([t])
        conf = cell(t, seeds=CONFIRM)
        print(f"  {t:28s} {ar.mean():5.1f}±{ar.std():4.1f} {a5.mean():6.1f} {b1.mean():6.1f} "
              f"{(f'{np.nanmedian(tt):6.0f}' if np.isfinite(tt).any() else '     -')} {int(np.isfinite(tt).sum())}/4 "
              f"{bp[t]:.4f}{'' if off else '*'}" + (f"   seeds 4-7: {conf.mean():5.1f}" if conf is not None else ""))
    best = {}
    for arm, lrs in grid[block].items():
        lrs = sorted(lrs, key=float)
        m = {lr: cell(tags[(arm, lr)]) for lr in lrs}
        if any(v is None for v in m.values()):
            print(f"  best lr {arm}: incomplete")
            continue
        top = max(v.mean() for v in m.values())
        near = [lr for lr in lrs if m[lr].mean() >= top - 2]
        bp, off = mean_bpb([tags[(arm, lr)] for lr in near])
        best[arm] = min(near, key=lambda lr: (bp[tags[(arm, lr)]], float(lr)))
        i = lrs.index(best[arm])
        nb = [lrs[j] for j in (i - 1, i + 1) if 0 <= j < len(lrs)]
        edge = len(lrs) > 1 and i in (0, len(lrs) - 1) and all(m[best[arm]].mean() > m[n].mean() + 2 for n in nb)
        k = max(range(len(lrs)), key=lambda j: m[lrs[j]].mean())   # area argmax
        soft = not edge and len(lrs) > 1 and k in (0, len(lrs) - 1)
        print(f"  best lr {arm}: {best[arm]} (grid {', '.join(lrs)}; within 2 of the top: {', '.join(near)}"
              f"{'' if len(near) == 1 or off else '; tie broken on the in-training bpb *'})"
              + (f"  OPEN BRACKET at the {'upper' if i else 'lower'} edge" if edge else "")
              + (f"  (note: area argmax {lrs[k]} at the {'upper' if k else 'lower'} edge by <= 2: not an open bracket "
                 f"by rule 1)" if soft else ""))
    if "none" in best:
        nb = tags[("none", best["none"])]
        for arm in ("win-ctx", "win-text"):
            if arm in best:
                t = tags[(arm, best[arm])]
                txt, ok, gap = test(t, nb)
                txt5, ok5, _ = test(t, nb, area5)
                off = mean_bpb([t, nb])[1]   # offline bpb if every run of both cells has it, else in-training
                cost = test(t, nb, lambda r: 1000 * bpb[(r["args"]["tag"][1:], r["args"]["seed"])][0 if off else 1])[0]
                ctxt, cok, cgap = test(t, nb, seeds=CONFIRM)
                if cgap is not None:   # confirm seeds 4-7: same sign, >= 3/4 wins (selection-free)
                    d = cell(t, seeds=CONFIRM) - cell(nb, seeds=CONFIRM)
                    cok = d.mean() > 0 and (d > 0).sum() >= 3
                    verdict = "SURVIVES tuning" if ok and cok else "does NOT survive tuning"
                else:
                    verdict = ("survival pending: confirm seeds 4-7 needed (gap < 10)" if ok and gap is not None and gap < 10
                               else "SURVIVES tuning" if ok else "does NOT survive tuning")
                print(f"  {t} - {nb}: area {txt}; without b11 {txt5}; LM cost ({'offline' if off else 'in-training*'} bpb "
                      f"x1e-3, arm - none; wins = seeds where the arm costs) {cost}"
                      + (f"\n    confirm seeds 4-7: area {ctxt} -> {'agrees' if cok else 'does NOT agree'}" if cgap is not None else "")
                      + f"\n    -> 2% speed-up {verdict}"
                      f"{'' if arm == 'win-ctx' else ' (text site)'}{'' if ok == ok5 else ' (differs without b11)'}")
    if block == "pt2long" and "none" in best and "win-ctx" in best:
        def headline(seeds, label):
            n_ = [runs[(tags[("none", best["none"])], s)] for s in seeds]
            w_ = [runs[(tags[("win-ctx", best["win-ctx"])], s)] for s in seeds]
            bud = budget(n_[0])
            tn = np.array([t90(r) if np.isfinite(t90(r)) else bud for r in n_])
            tw = np.array([t90(r) if np.isfinite(t90(r)) else bud for r in w_])
            idx = np.random.default_rng(0).integers(0, len(tn), size=(10000, len(tn)))
            rb = np.median(tn[idx], 1) / np.median(tw[idx], 1)
            cen_b = np.mean([censored_median(tn[i], bud) for i in idx])
            cen_n, cen_w = int((tn >= bud).sum()), int((tw >= bud).sum())
            ge = ">=" if censored_median(tn, bud) else ""
            per = n_[0]["args"]["bs"] * n_[0]["args"]["seq"] / 1e6
            ev = w_[0].get("prior_events") or []
            win = f"window {ev[0][0] * per:.0f}-{ev[1][0] * per:.0f}M tokens" if len(ev) == 2 else "window ?"
            ratio = ("no ratio (the win-ctx median is censored)" if censored_median(tw, bud) else
                     f"{ge}{np.median(tn) / np.median(tw):.2f}x (90% CI {np.percentile(rb, 5):.2f}-{np.percentile(rb, 95):.2f}"
                     f"{f'; the none median is censored in {100 * cen_b:.0f}% of the resamples' if cen_b else ''})")
            print(f"  {label} t90 none@{best['none']} / win-ctx@{best['win-ctx']}: {ratio}; medians {ge}{np.median(tn):.0f}M / "
                  f"{np.median(tw):.0f}M; censored at {bud:.0f}M: none {cen_n}/{len(seeds)}, win-ctx {cen_w}/{len(seeds)}; "
                  f"{win} of {bud:.0f}M (this budget's schedule)")
        headline(MAIN, "HEADLINE (seeds 0-3)")
        if all((tags[(a, best[a])], s) in runs for a in ("none", "win-ctx") for s in CONFIRM):
            headline(CONFIRM, "confirm (seeds 4-7)")
        if ("win-shuf-ctx", "1e-3") in tags and ("win-ctx", "1e-3") in tags:
            x, y = cell(tags[("win-ctx", "1e-3")]), cell(tags[("win-shuf-ctx", "1e-3")])
            if x is None or y is None:
                print("  CONTROL win-ctx@1e-3 - win-shuf-ctx@1e-3: incomplete")
            else:
                d = x - y
                print(f"  CONTROL win-ctx@1e-3 - win-shuf-ctx@1e-3: {d.mean():+.1f} (wins {int((d > 0).sum())}/4) -> "
                      f"{'passes (shuffled geometry below the helix)' if d.mean() > 0 and (d > 0).sum() >= 3 else 'FAILS: geometry specificity not shown'}")
