"""Summary of the rainbow pretraining runs (tools/pretrain_rainbow.py; runs/ptrainbow): per arm and task (add / days /
months), held-out accuracy at the end, the curve mean (area), first step the held-out accuracy reaches 0.5 / 0.9 (stays
above), and paired differences to none (same seed = same init and data stream; paired t-test). Also the final
val loss and the CKA of the eval states to each true geometry.

    python tools/ptrainbow_summary.py <out dir>
"""
import collections, glob, json, sys
import numpy as np
from scipy import stats

def stays(cur, key, thr):
    v = [e[key] for e in cur]
    for i in range(len(v)):
        if all(x >= thr for x in v[i:]):
            return cur[i]["tokens"] / 1e6
    return np.nan

runs = collections.defaultdict(dict)
for f in glob.glob(f"{sys.argv[1]}/run_*.json"):
    r = json.load(open(f))
    if r["curve"][-1]["step"] < r["steps"] - 1:   # audit 2026-10-14: never read a run that is still going
        print("unfinished, skipped:", f.rsplit("/", 1)[-1]); continue
    runs[r["args"]["priors"]][r["args"]["seed"]] = r
order = ["none", "helix", "helix_shuf", "days", "months", "helix+days+months", "helix+days+months@norm", "helix+days+months@bal",
         "days+months", "helix+months", "helix+days", "helix_shuf+days_shuf+months_shuf"]
base = runs["none"]
for task in ("add", "days", "months"):
    key = f"{task}_held"
    print(f"\n== {task}: held-out (never-trained {'pairs' if task == 'add' else 'item x shift classes'}); paired tests on the seeds both arms have")
    print(f"  {'arm':34s} {'final':>13s} {'area':>12s} {'Δarea vs none (paired p)':>26s} {'t50 / t90 (M tok, median)':>26s}")
    for arm in order:
        rs = runs.get(arm, {})
        if not rs:
            continue
        seeds = sorted(rs)
        fin = [100 * rs[s]["curve"][-1][key] for s in seeds]
        area = [100 * np.mean([e[key] for e in rs[s]["curve"]]) for s in seeds]
        t50 = np.nanmedian([stays(rs[s]["curve"], key, 0.5) for s in seeds]) if any(np.isfinite([stays(rs[s]["curve"], key, 0.5) for s in seeds])) else np.nan
        t90v = [stays(rs[s]["curve"], key, 0.9) for s in seeds]
        t90 = np.nanmedian(t90v) if any(np.isfinite(t90v)) else np.nan
        d = ""
        if arm != "none":
            common = [x for x in seeds if x in base]
            a_ = [100 * np.mean([e[key] for e in rs[x]["curve"]]) for x in common]
            b = [100 * np.mean([e[key] for e in base[x]["curve"]]) for x in common]
            diff = np.array(a_) - np.array(b)
            d = f"{diff.mean():+6.1f} (p {stats.ttest_rel(a_, b).pvalue:.3f}, n={len(common)})" if len(common) >= 3 else "n<3"
        print(f"  {arm:34s} {np.mean(fin):5.1f} [{min(fin):3.0f}-{max(fin):3.0f}] {np.mean(area):5.1f} ±{np.std(area):4.1f} {d:>32s} "
              f"{t50:8.0f} / {t90:6.0f} ({int(np.isfinite(t90v).sum())}/{len(seeds)})")
print("\n== val loss (final, mean over seeds) and CKA of the eval states to the true geometries (layer 4, final)")
for arm in order:
    rs = runs.get(arm, {})
    if rs:
        e = [rs[s]["curve"][-1] for s in sorted(rs)]
        print(f"  {arm:34s} val {np.mean([x['val_loss'] for x in e]):.4f}  CKA helix {np.mean([x['cka_helix@L4'] for x in e]):.2f} "
              f"days {np.mean([x['cka_days@L4'] for x in e]):.2f} months {np.mean([x['cka_months@L4'] for x in e]):.2f}")
