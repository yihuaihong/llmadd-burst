"""The ONE definition set of the Batch 1 read-outs (frozen 2026-10-16, before any Batch 1 result existed), imported by
tools/pt2main_summary.py (Burst pt2main, report/audits/pt2main_AUDIT.md) and tools/ptscale_summary.py (Torch runs/ptlr
+ its reference cells, runs/ptlr/AUDIT.md); tools/xsite_summary.py (report/audits/burst_infra_AUDIT.md) uses its area,
divergence and Burst run selection. Both AUDIT files point here (amendments of 2026-10-16). tools/pt999_summary.py keeps
its own area / t90 code on purpose: the summaries check their cell areas against its printed ones (double path).

  area(r, bs)          100 x mean over ALL evals of the mean held-out exact match over the buckets bs (default the 6;
                       B5 = without b11, whose held pairs occur in the natural text)
  t_theta(r, th, bs)   tokens (M) until the bucket mean first reaches th, linear interpolation between evals; nan if never
                       (= censored at the budget, budget_M(r))
  diverged_at(r)       step of the first eval with a non-finite lm, val_loss, geo or grad_prior (None if none). A diverged
                       run is data: it keeps its area (it counts against its lr) and its bpb counts as worst.
  unstable(r, ref)     final val_loss > ref's final val_loss + 0.5 nats (ref = the same seed's none at the base lr):
                       flagged 'unstable', kept like any run
  best_lr(...)         argmax of the mean area over seeds 0-3; tied lrs -> the lower mean bpb, offline (full val) if every
                       run of every tied lr has it, else the in-training bpb for all of them (marked '*'; never a mix); a
                       non-finite bpb is the worst; an exact bpb tie -> the lower lr. Tied = every lr within 2 area points
                       of the top (near='within': pt2main_AUDIT rule 1) or the top two if within 2 (near='top2':
                       runs/ptlr/AUDIT.md rule 1 as pre-registered)
  bracket(...)         'open' = the best lr is at an edge of the grid and beats its neighbour by > 2 points; an area argmax
                       at the edge by <= 2 is a note only
  survives(...)        best-vs-best area difference > 0, >= 3/4 seed wins and paired t p < .05 on seeds 0-3, and the confirm
                       seeds 4-7 (same sign, >= 3/4 wins; selection-free) when they exist. Without them: 'survives
                       (unconfirmed)' (runs/ptlr/AUDIT.md rule 3); with confirm_free_gap=10 the pt2main_AUDIT rule 2 as
                       pre-registered: 'SURVIVES' for a gap >= 10, 'survival pending' below (the shared reading is then
                       returned as 'proposed', pending the user's sign-off)
  speedup(...)         "N x" = median t(reference arm) / median t(prior arm) over the seeds; a run that never reaches theta
                       counts as +inf. A median of n values is censored when >= (n + 1) // 2 of them are (4 seeds: 2);
                       its value with the censored runs at the budget is then a lower bound. Censored reference median ->
                       '>=' (the reference is the slower arm); censored prior median -> '<=' (the prior is the slower arm;
                       the slower / faster ratio is again a lower bound; arm_censored='none' gives pt2main_AUDIT rule 3 as
                       pre-registered: no ratio, the '<=' value returned as 'proposed'); both censored (the faster arm's
                       median is censored) -> no ratio. With every censored run at one common time (the budget, beyond all
                       observed crossings) this median equals the Kaplan-Meier median (km_median, midpoint convention
                       where S = 0.5 on an interval) whenever it is not censored: asserted on every call.
  boot_ci(...)         paired seed bootstrap (10000 resamples of the seed indices, numpy default_rng(0)), 90% percentile
                       CI of the ratio with censored runs at the budget, the shares of resamples with a censored median;
                       printed with 'n=4: indicative only'
  pick_burst_runs(...) Burst re-runs (rule 6): per (tag, seed) the first finished run in task order whose check passed,
                       else the first finished one (its failed check reported); the others are reported
  signflip_p(d)        exact two-sided paired sign-flip p (all 2^n sign patterns; = ptgw_summary.signflip_p)
  check_2pct(r, arm, none_args)  validity of a Torch pt999 2% run (2pct-none / -gate-ctx / -win-ctx = Batch 1 T2), the
                       runs/ptlr/AUDIT.md 'Controls': code_sha 79e9b410dcc3, 18310 steps / 75 evals, args = the same seed's
                       2pct-none args except target / schedule (and out / tag), step-0 lm vs the 2% values, window
                       prior_on_frac 6408 / 18310, none never on; used by xsite_summary (P1b) and ptscale_summary (T2)
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

B = ("b11", "b12", "b13", "b22", "b23", "b33")
B5 = B[1:]                    # without b11
MAIN, CONFIRM = (0, 1, 2, 3), (4, 5, 6, 7)
TIE = 2.0                     # area points: lrs within TIE of the top are tied (best_lr), and the bracket margin
UNSTABLE_NATS = 0.5
N_BOOT = 10000


def held(e, bs=B):
    return np.mean([e[f"held_{b}"] for b in bs])


def area(r, bs=B):
    return 100 * np.mean([held(e, bs) for e in r["curve"]])


def t_theta(r, theta=0.9, bs=B):
    """M tokens until the bucket mean first reaches theta (linear interpolation between evals); nan if never"""
    prev, per_step = None, r["args"]["bs"] * r["args"]["seq"]
    for e in r["curve"]:
        v, tk = held(e, bs), e["step"] * per_step
        if v >= theta:
            if prev is None:
                return tk / 1e6
            t0, v0 = prev
            return (t0 + (theta - v0) / (v - v0) * (tk - t0)) / 1e6
        prev = (tk, v)
    return np.nan


def budget_M(r):
    return r["steps"] * r["args"]["bs"] * r["args"]["seq"] / 1e6


def finished(r):
    return r["curve"][-1]["step"] >= r["steps"] - 1


def diverged_at(r):
    fin = lambda v: v is None or math.isfinite(v)
    bad = [e["step"] for e in r["curve"]
           if not (fin(e.get("lm")) and fin(e.get("val_loss")) and fin(e.get("geo")) and fin(e.get("grad_prior")))]
    return bad[0] if bad else None


def unstable(r, ref):
    """final val_loss above the reference run's by more than 0.5 nats (False when either is not finite or ref is None)"""
    if ref is None:
        return False
    v, w = r["curve"][-1].get("val_loss"), ref["curve"][-1].get("val_loss")
    return v is not None and w is not None and math.isfinite(v) and math.isfinite(w) and v > w + UNSTABLE_NATS


def offline_bpb(entry, diverged=False):
    """bpb_full of an offline_lm.py entry if it is a full-val one whose double path holds (or the run diverged), else None"""
    if not entry or entry.get("max_windows", 0) != 0:
        return None
    if not diverged and not abs(entry.get("orig_minus_logged", math.inf)) < 0.002:
        return None
    return entry["bpb_full"]


def worst(v):
    return v if v is not None and math.isfinite(v) else math.inf


def best_lr(cells, near_rule="within"):
    """cells: {lr (str): {"area": [seed values], "off": [offline bpb or None], "intr": [in-training bpb]}} (seeds 0-3).
    near_rule 'within': every lr within TIE of the top is tied; 'top2': only the top two, if within TIE.
    -> {"best", "near", "offline" (bool: the tie-break used offline bpb), "mean": {lr: mean area}, "bpb": {lr: mean}}"""
    lrs = sorted(cells, key=float)
    m = {lr: float(np.mean(cells[lr]["area"])) for lr in lrs}
    top = max(m.values())
    if near_rule == "top2":
        order = sorted(lrs, key=lambda lr: (-m[lr], float(lr)))
        near = sorted(order[:2] if len(order) > 1 and m[order[0]] - m[order[1]] <= TIE else order[:1], key=float)
    else:
        assert near_rule == "within", near_rule
        near = [lr for lr in lrs if m[lr] >= top - TIE]
    off = all(v is not None for lr in near for v in cells[lr]["off"])
    bp = {lr: float(np.mean([worst(v) for v in cells[lr]["off" if off else "intr"]])) for lr in near}
    best = min(near, key=lambda lr: (bp[lr], float(lr)))
    return {"best": best, "near": near, "offline": off, "mean": m, "bpb": bp}


def bracket(lrs, mean, best):
    """lrs sorted; -> ('open', 'upper' | 'lower') if best is at an edge and beats its neighbour by > TIE,
    ('note', edge) if the area argmax is at an edge by <= TIE, else None"""
    if len(lrs) < 2:
        return None
    i = lrs.index(best)
    nb = [lrs[j] for j in (i - 1, i + 1) if 0 <= j < len(lrs)]
    if i in (0, len(lrs) - 1) and all(mean[best] > mean[n] + TIE for n in nb):
        return "open", "upper" if i else "lower"
    k = max(range(len(lrs)), key=lambda j: mean[lrs[j]])
    if k in (0, len(lrs) - 1):
        return "note", "upper" if k else "lower"
    return None


def paired(x, y):
    """x - y over the same seeds -> (mean diff, p of the paired t-test (nan if the differences are constant), wins, n)"""
    d = np.asarray(x, float) - np.asarray(y, float)
    p = stats.ttest_rel(x, y).pvalue if len(d) > 1 and np.std(d) > 0 else float("nan")
    return float(d.mean()), float(p), int((d > 0).sum()), len(d)


def fmt_paired(t):
    g, p, w, n = t
    return f"{g:+6.1f} (p {p:.3f}, wins {w}/{n})"


def survives(x, y, xc=None, yc=None, confirm_free_gap=None):
    """best-vs-best: arm x minus reference y on seeds 0-3 (xc, yc: the confirm seeds 4-7, or None while missing).
    confirm_free_gap: pt2main_AUDIT rule 2 as pre-registered (without confirm seeds a gap >= it survives, below it
    'survival pending'); the shared verdict is then also returned as "proposed" when it differs.
    -> {"main": paired tuple, "main_ok", "confirm": paired tuple or None, "confirm_ok", "verdict", "proposed"}"""
    t = paired(x, y)
    g, p, w, n = t
    ok = g > 0 and w >= 0.75 * n and p < .05
    out = {"main": t, "main_ok": ok, "confirm": None, "confirm_ok": None}
    if xc is not None and yc is not None:
        c = paired(xc, yc)
        out["confirm"], out["confirm_ok"] = c, c[0] > 0 and c[2] >= 0.75 * c[3]
    if not ok:
        out["verdict"] = "does NOT survive tuning"
    elif out["confirm"] is None:
        out["verdict"] = "survives (unconfirmed): a paper claim needs the confirm seeds 4-7"
    elif out["confirm_ok"]:
        out["verdict"] = "SURVIVES tuning (confirmed on seeds 4-7)"
    else:
        out["verdict"] = "does NOT survive tuning (the confirm seeds 4-7 disagree)"
    out["proposed"] = None
    if confirm_free_gap is not None and ok and out["confirm"] is None:
        out["proposed"] = out["verdict"]
        out["verdict"] = (f"SURVIVES tuning (gap >= {confirm_free_gap:g}: no confirm seeds needed by the pre-registered rule)"
                          if g >= confirm_free_gap else
                          f"survival pending (gap < {confirm_free_gap:g}: the confirm seeds 4-7 decide)")
    return out


def censored_median(t):
    """t: crossing times, nan / inf = censored. True when the median includes a censored run"""
    t = np.asarray(t, float)
    return int((~np.isfinite(t)).sum()) >= (len(t) + 1) // 2


def median_at_budget(t, bud):
    t = np.asarray(t, float)
    return float(np.median(np.where(np.isfinite(t), t, bud)))


def km_median(t):
    """Kaplan-Meier median of crossing times t (nan / inf = censored, all at one time beyond every crossing); midpoint
    convention where the curve sits at exactly 0.5 between two events; nan if it never drops to 0.5 or sits at 0.5 up
    to the censoring time (conventions differ there)"""
    t = np.asarray(t, float)
    ev = np.sort(t[np.isfinite(t)])
    S, at_risk, times, surv = 1.0, len(t), [], []
    for u in np.unique(ev):
        d = int((ev == u).sum())
        S *= 1 - d / at_risk
        at_risk -= d
        times.append(u); surv.append(S)
    for i, (u, s) in enumerate(zip(times, surv)):
        if s < 0.5 - 1e-12:
            return float(u)
        if abs(s - 0.5) <= 1e-12:
            return float((u + times[i + 1]) / 2) if i + 1 < len(times) else float("nan")
    return float("nan")


def speedup(t_ref, t_arm, bud, arm_censored="bound"):
    """'N x' of the arm over the reference (t: per-seed crossing times in M tokens, nan = never; bud = budget in M).
    arm_censored 'bound': a censored arm median gives '<='; 'none': no ratio (pt2main_AUDIT rule 3 as pre-registered),
    the '<=' text returned as "proposed".
    -> {"ratio": float or None, "mark": '' | '>=' | '<=', "med_ref", "med_arm" (censored runs at the budget),
        "cen_ref", "cen_arm" (counts), "text", "proposed"}"""
    assert arm_censored in ("bound", "none"), arm_censored
    for t in (t_ref, t_arm):   # double path: the censored-at-budget median = the KM median whenever it is not censored
        if not censored_median(t):
            assert abs(km_median(t) - median_at_budget(t, bud)) < 1e-9, (t, km_median(t), median_at_budget(t, bud))
    cr, ca = censored_median(t_ref), censored_median(t_arm)
    mr, ma = median_at_budget(t_ref, bud), median_at_budget(t_arm, bud)
    nr, na = int((~np.isfinite(np.asarray(t_ref, float))).sum()), int((~np.isfinite(np.asarray(t_arm, float))).sum())
    out = {"med_ref": mr, "med_arm": ma, "cen_ref": nr, "cen_arm": na, "ratio": None, "mark": "", "proposed": None}
    if cr and ca:
        out["text"] = "no ratio (both medians censored: the faster arm's median is censored)"
        return out
    out["ratio"], out["mark"] = mr / ma, ">=" if cr else "<=" if ca else ""
    out["text"] = f"{out['mark']}{out['ratio']:.2f}x" + (" (the prior arm is the slower one; its median is censored)" if ca else "")
    if ca and arm_censored == "none":
        out["proposed"], out["ratio"], out["mark"] = out["text"], None, ""
        out["text"] = "no ratio (the prior arm's median is censored)"
    return out


def boot_ci(t_ref, t_arm, bud, n_boot=N_BOOT):
    """paired seed bootstrap of the ratio (censored runs at the budget) -> text with the 90% CI and the censored shares"""
    fx, fy = np.isfinite(np.asarray(t_ref, float)), np.isfinite(np.asarray(t_arm, float))
    x, y = np.where(fx, t_ref, bud), np.where(fy, t_arm, bud)
    n = len(x)
    idx = np.random.default_rng(0).integers(0, n, size=(n_boot, n))
    rb = np.median(x[idx], 1) / np.median(y[idx], 1)
    cx = float(np.mean((~fx)[idx].sum(1) >= (n + 1) // 2))   # resamples whose median includes a censored run
    cy = float(np.mean((~fy)[idx].sum(1) >= (n + 1) // 2))
    lo, hi = np.percentile(rb, 5), np.percentile(rb, 95)
    extra = "".join([f"; the reference median is censored in {100 * cx:.0f}% of the resamples" if cx else "",
                     f"; the prior median is censored in {100 * cy:.0f}% of the resamples" if cy else ""])
    return {"lo": float(lo), "hi": float(hi), "cen_ref": float(cx), "cen_arm": float(cy),
            "text": f"90% CI {lo:.2f}-{hi:.2f} (paired seed bootstrap{extra}; n={n}: indicative only)"}


def best_lr_t90(med, mean_area):
    """lr with the lowest median t90 (censored runs at the budget); ties -> the higher mean area, then the lower lr"""
    return min(med, key=lambda lr: (med[lr], -mean_area[lr], float(lr)))


def pick_burst_runs(res, tasks):
    """Burst results dir + [(task name, tag, seed)] in task order -> (runs {(tag, seed): (task dir, run)}, problems).
    A re-run supersedes an unfinished or missing run, or a finished run whose check failed; of several finished runs
    the first whose check passed counts, else the first; everything else is reported."""
    found = {}
    for name, tag, s in tasks:
        task = Path(res) / name
        fs = sorted(task.glob(f"{tag}/run_*.json"))
        if not fs:
            found.setdefault((tag, s), []).append((task, "missing", None))
            continue
        assert len(fs) == 1, f"two runs in {task}: {fs}"
        r = json.loads(fs[0].read_text())
        assert r["args"]["tag"] == f"_{tag}" and r["args"]["seed"] == s, (fs[0], r["args"]["tag"], r["args"]["seed"])
        st = "finished" if finished(r) else f"unfinished (last eval {r['curve'][-1]['step']} of {r['steps']})"
        found.setdefault((tag, s), []).append((task, st, r))
    runs, problems = {}, []
    for (tag, s), L in found.items():
        fin = sorted([(task, r) for task, st, r in L if st == "finished"], key=lambda x: not check_passed(x[0]))
        for task, st, _ in L:
            if st != "finished":
                problems.append(f"{st}: {task.name}" + (f" (superseded by {fin[0][0].name})" if fin else ""))
        if not fin:
            continue
        runs[(tag, s)] = fin[0]
        if len(fin) > 1:
            problems.append(f"{len(fin)} finished runs of {tag} seed {s}: {fin[0][0].name} counts (the first whose check "
                            f"passed, else the first), {', '.join(t.name for t, _ in fin[1:])} ignored")
    return runs, problems


def check_passed(task, name="check.json", key="pass"):
    f = Path(task) / name
    return f.exists() and bool(json.loads(f.read_text()).get(key))


def signflip_p(d):
    """exact two-sided paired sign-flip p over all 2^n sign patterns (= tools/ptgw_summary.py signflip_p)"""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    return float(np.mean([abs(np.mean(d * np.array(sg))) >= obs - 1e-12 for sg in itertools.product((1, -1), repeat=len(d))]))


# ---------------------------------------------------------------- 2% references (Torch pt999 snapshot)
SNAP_PT999 = "79e9b410dcc3"
STEPS_3E8, EVALS_3E8 = 18310, 75
W_EVENTS = [[916, 1], [7324, 0]]
W_ON = 6408 / 18310           # window (0.05, 0.4): prior on at steps 916..7323
# step-0 lm of the 2% seeds 0-7 (runs/ptlr/AUDIT.md 'Controls'; lr- and arm-independent)
LM0_2PCT = (11.61647, 11.58723, 11.56547, 11.60211, 11.59495, 11.56812, 11.58452, 11.59349)
ARM_2PCT = {"none": ("none", "whole"), "gate-ctx": ("helix3", "gate"), "win-ctx": ("helix3", "window")}
FIXED_2PCT = {"arith_lines": 1, "lr": 1e-3, "d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1536, "bs": 32, "seq": 512,
              "tokens": 3e8, "lam": 5.0, "site": "ctx", "window": "0.05,0.4", "tokenization": "whole", "eval_every": 250,
              "n_eval": 150}


def check_2pct(r, arm, none_args):
    """reasons why a Torch pt999 2% run of this arm is not a valid reference (empty list = valid); none_args = the args of
    the same seed's 2pct-none run (for arm 'none': its own)"""
    a, c, bad = r["args"], r["curve"], []
    want_t, want_s = ARM_2PCT[arm]
    if r.get("code_sha") != SNAP_PT999:
        bad.append(f"code_sha {r.get('code_sha')} (want {SNAP_PT999})")
    if not (r["steps"] == STEPS_3E8 and len(c) == EVALS_3E8 and c[-1]["step"] == STEPS_3E8 - 1):
        bad.append(f"steps {r['steps']} / {len(c)} evals / last eval {c[-1]['step']} (want {STEPS_3E8} / {EVALS_3E8} / {STEPS_3E8 - 1})")
    if r.get("grad_ckpt_on") not in (None, False):
        bad.append(f"grad_ckpt_on {r.get('grad_ckpt_on')} at 53M")
    if (a.get("target"), a.get("schedule")) != (want_t, want_s):
        bad.append(f"target / schedule {a.get('target')} / {a.get('schedule')} (want {want_t} / {want_s})")
    diff = sorted(k for k in set(a) | set(none_args) if k not in ("out", "tag", "target", "schedule")
                  and a.get(k, "<absent>") != none_args.get(k, "<absent>"))
    if diff:
        bad.append(f"args differ from the same seed's 2pct-none in {diff}")
    wrong = {k: a.get(k) for k, v in FIXED_2PCT.items() if a.get(k) != v}
    if wrong:
        bad.append(f"args {wrong} (want {({k: FIXED_2PCT[k] for k in wrong})})")
    s = a.get("seed")
    if not (isinstance(s, int) and 0 <= s < len(LM0_2PCT) and math.isfinite(c[0]["lm"]) and abs(c[0]["lm"] - LM0_2PCT[s]) <= 1e-3):
        bad.append(f"step-0 lm {c[0]['lm']} vs the 2% seed-{s} value {LM0_2PCT[s] if isinstance(s, int) and 0 <= s < 8 else '?'}")
    on = c[-1].get("prior_on_frac")
    if arm == "win-ctx":
        if on is None or abs(on - W_ON) > 1e-6:
            bad.append(f"prior_on_frac {on} (want {W_ON:.6f})")
        if "prior_events" in r and r["prior_events"] != W_EVENTS:
            bad.append(f"prior_events {r['prior_events']}")
    if arm == "none" and (c[-1].get("prior_used_frac", 0) != 0 or (on or 0) != 0):
        bad.append(f"none run with prior_on_frac {on} / prior_used_frac {c[-1].get('prior_used_frac')}")
    return bad
