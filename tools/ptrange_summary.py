"""Summary of experiment B (tools/pretrain_range.py; operands 0..999): per tokenisation x target, final held-out exact
match per operand-length bucket, the never-trained 4-digit buckets, the curve mean over buckets (area), paired
difference to none (same seed = same init, data stream and held-out split), and val bits/byte.

    python tools/ptrange_summary.py <out dir> [<out dir> ...]
"""
import collections, glob, json, sys
import numpy as np
from scipy import stats

B = ("b11", "b12", "b13", "b22", "b23", "b33")
runs = collections.defaultdict(dict)
for d in sys.argv[1:]:
    for f in glob.glob(f"{d}/*/run_*.json"):
        r = json.load(open(f)); a = r["args"]
        name = f"{a['tokenization']}-{a['target']}" + (f"-{a['schedule']}" if a.get("schedule", "whole") != "whole" else "")
        if r["curve"][-1]["step"] < r["steps"] - 1:
            print("unfinished:", f); continue
        runs[name][a["seed"]] = r
held = lambda e: np.mean([e[f"held_{b}"] for b in B])
for tk in ("whole", "digits"):
    base = runs.get(f"{tk}-none", {})
    print(f"\n== {tk} tokens: final held-out exact match per bucket | 4-digit (never trained) | area (mean over buckets and evals) | bits/byte")
    for name in sorted(k for k in runs if k.startswith(tk)):
        rs = runs[name]; s = sorted(rs)
        fin = {b: np.mean([100 * rs[x]["curve"][-1][f"held_{b}"] for x in s]) for b in B}
        ood = np.mean([100 * np.mean([rs[x]["curve"][-1][k] for k in ("ood_b44", "ood_b41", "ood_b42")]) for x in s])
        area = [100 * np.mean([held(e) for e in rs[x]["curve"]]) for x in s]
        bpb = [rs[x]["curve"][-1]["val_bits_per_byte"] for x in s]
        d = ""
        if base and not name.endswith("-none"):
            common = [x for x in s if x in base]
            ba = [100 * np.mean([held(e) for e in base[x]["curve"]]) for x in common]
            aa = [100 * np.mean([held(e) for e in rs[x]["curve"]]) for x in common]
            bb = [base[x]["curve"][-1]["val_bits_per_byte"] for x in common]
            d = (f" Δarea {np.mean(aa) - np.mean(ba):+5.1f} (p {stats.ttest_rel(aa, ba).pvalue:.3f}) "
                 f"Δbpb {np.mean(np.array(bpb)[[s.index(x) for x in common]] - np.array(bb)):+.4f}")
        print(f"  {name:32s} n={len(s)} " + " ".join(f"{b}:{fin[b]:3.0f}" for b in B) + f" | 4dig {ood:3.0f} | area {np.mean(area):5.1f}±{np.std(area):4.1f} | bpb {np.mean(bpb):.4f}{d}")
