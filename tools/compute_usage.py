"""GPU-hours used by this project, for the paper's Compute paragraph.

Burst: status/<task>.json (elapsed_min, kind) x the task's '# gpus: N' header (A100 40GB).
Torch: sacct records of llmadd_* jobs (elapsed x gres/gpu, partition a100_tandon / h100_tandon), passed as a file:
    sacct -u $USER -S 2026-09-01 -X -n -P --format=JobName,State,ElapsedRaw,AllocTRES,Partition | grep '^llmadd_' > torch.txt
    python tools/compute_usage.py torch.txt
Failed and cancelled jobs are counted too (they used the GPU); queued or still-running jobs are not.
"""

from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# experiment family by name (first match wins); names are task names on Burst and job names on Torch
FAMILIES = [
    ("from-scratch pretraining", r"^(pt|ptdiag)"),
    ("toy grokking", r"toy"),
    ("general-ability evals", r"^ge-"),
    ("learning curves (Fig. 1)", r"curve|^fig1|^b3-|^s1-|^long"),
    ("concept domains (7-domain window, months / weekdays)", r"concept|months|days|route|p2|p3-|e2-outctrl|e5"),
    ("multiplication / mixed ops / structure", r"mul|mixed|mixops|factor|^dm-|^b1-|mod|max"),
    ("regularity dose / self-timed prior", r"^reg|^gate"),
    ("geometry diagnostics (kappa, spike)", r"geomdiag|diag"),
    ("component ablations (sep / nonsep / coarse)", r"p4-sep|kgate|e3-"),
    ("data volume x checkpoint", r"e1-size|322|323|324"),
    ("model downloads / setup / smoke tests", r"hello|env|models|dl|smoke|fetch"),
]


def family(name: str) -> str:
    n = re.sub(r"^\d+[a-z]?-", "", name.removeprefix("llmadd_"))
    for fam, pat in FAMILIES:
        if re.search(pat, n):
            return fam
    return "fine-tuning sweeps (sizes, checkpoints, seeds, methods)"


def burst() -> dict:
    out = collections.defaultdict(float)
    for p in glob.glob(str(REPO / "status" / "*.json")):
        s = json.loads(Path(p).read_text())
        if s.get("kind") != "gpu":
            continue
        sh = REPO / "tasks" / f"{s['task']}.sh"
        m = re.search(r"^# gpus: (\d+)", sh.read_text(), re.M) if sh.exists() else None
        out[family(s["task"])] += float(s.get("elapsed_min") or 0) / 60 * (int(m.group(1)) if m else 1)
    return out


def torch(path: str) -> dict:
    out = collections.defaultdict(lambda: collections.defaultdict(float))
    for line in Path(path).read_text().splitlines():
        name, state, sec, tres, part = line.split("|")[:5]
        g = re.search(r"gres/gpu=(\d+)", tres)
        if not g or state in ("PENDING", "RUNNING"):
            continue
        kind = "H100" if "h100" in part else "A100" if "a100" in part else part
        out[family(name)][kind] += int(sec) / 3600 * int(g.group(1))
    return out


if __name__ == "__main__":
    b = burst()
    t = torch(sys.argv[1]) if len(sys.argv) > 1 else {}
    fams = sorted(set(b) | set(t), key=lambda f: -(b.get(f, 0) + sum(t.get(f, {}).values())))
    print(f"| family | Burst A100-40GB | Torch A100-80GB | Torch H100 | total GPU-h |\n|---|---|---|---|---|")
    tot = collections.Counter()
    for f in fams:
        row = (b.get(f, 0), t.get(f, {}).get("A100", 0), t.get(f, {}).get("H100", 0))
        tot.update({"b": row[0], "a": row[1], "h": row[2]})
        print(f"| {f} | {row[0]:.1f} | {row[1]:.1f} | {row[2]:.1f} | {sum(row):.1f} |")
    print(f"| **total** | {tot['b']:.1f} | {tot['a']:.1f} | {tot['h']:.1f} | {sum(tot.values()):.1f} |")
