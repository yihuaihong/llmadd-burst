"""Summary tables for Torch runs (runs/<name>/out or results/<name>-torch).

  python tools/report_tables.py arms <name> [<name> ...]     each arm vs none, paired over seeds, per eval key
  python tools/report_tables.py general <name> [<name> ...]  base vs arms on HellaSwag / ARC-E / ARC-C / LAMBADA + LM loss
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RUNS = Path("/scratch/yh6210/llmadd_torch/runs")
REPO = Path(__file__).resolve().parents[1]
KEYS = ("test", "test_sub", "test_mul", "gpt3_unseen", "gpt3_unseen_add", "gpt3_unseen_sub", "gpt3_unseen_mul")
GEN = ("hellaswag", "arc_easy", "arc_challenge", "lambada", "mean")


def out_dir(name: str) -> Path:
    d = RUNS / name / "out"
    return d if d.exists() else REPO / "results" / f"{name}-torch"


def load(name: str) -> dict:
    runs = {}
    for p in glob.glob(str(out_dir(name) / "run_*_seed*.json")):
        r = json.loads(Path(p).read_text())
        runs.setdefault(r["geom"], {})[r["seed"]] = r
    return runs


def arms(name: str) -> None:
    runs = load(name)
    if "none" not in runs:
        print(name, "no none runs yet"); return
    first = runs["none"][min(runs["none"])]["eval"]
    print(f"== {name}  (none seeds {sorted(runs['none'])})")
    for k in (k for k in KEYS if k in first):
        cells = []
        for G in sorted(runs, key=lambda g: (g != "none", g)):
            s = sorted(set(runs[G]) & set(runs["none"]))
            a = np.array([runs[G][x]["eval"][k] for x in s]); b = np.array([runs["none"][x]["eval"][k] for x in s])
            if G == "none":
                cells.append(f"none {a.mean():.3f}")
            else:
                p = stats.ttest_rel(a, b).pvalue if len(s) > 1 else float("nan")
                cells.append(f"{G} {100 * (a - b).mean():+.1f} (p {p:.3f}, n {len(s)})")
        print(f"  {k:16s} " + " | ".join(cells))


def general(name: str) -> None:
    d = out_dir(name)
    try:
        b = json.loads((d / "base.json").read_text())["eval"]
    except FileNotFoundError:
        print(name, "no base yet"); return
    print(f"== {name}")
    g = b.get("general", {})
    print("  base           " + " ".join(f"{k} {100 * g[k]:.1f}" for k in GEN if k in g) + f" | lm {b.get('lm_val_loss', float('nan')):.3f}")
    for G, by_seed in sorted(load(name).items(), key=lambda kv: (kv[0] != "none", kv[0])):
        ev = list(by_seed.values())
        ev = [r["eval"] for r in ev]
        gen = {k: np.mean([e["general"][k] for e in ev]) for k in GEN} if all("general" in e for e in ev) else {}
        lm = np.mean([e["lm_val_loss"] for e in ev if "lm_val_loss" in e]) if any("lm_val_loss" in e for e in ev) else float("nan")
        print(f"  {G:16s} n={len(ev)} " + " ".join(f"{k} {100 * gen[k]:.1f}" for k in GEN if k in gen) +
              f" | lm {lm:.3f} | task test {100 * np.mean([e['test'] for e in ev]):.1f}")


if __name__ == "__main__":
    fn = {"arms": arms, "general": general}[sys.argv[1]]
    for n in sys.argv[2:]:
        fn(n)
