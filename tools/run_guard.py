"""Run guard for training jobs queued more than once (2026-10-14: the same run queued under two accounts /
partitions, because Slurm here does not let users move or cancel-and-replace pending jobs).

claim(path) -> True when this job should compute the run whose result file is `path`:
  - the result is already complete (curve reaches the last step)      -> False (nothing to do)
  - another job that is RUNNING right now has claimed it (<path>.claim) -> False (leave it to that job)
  - otherwise this job writes its id into <path>.claim and runs        -> True
A requeued job keeps its id and continues; a claim of a job that is no longer running is taken over. Training code
paths are unchanged: the guard only decides whether to start."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def complete(path) -> bool:
    try:
        r = json.loads(Path(path).read_text())
        return r["curve"][-1]["step"] >= r["steps"] - 1
    except Exception:
        return False


def _running(jobid: str) -> bool:
    try:
        o = subprocess.run(["squeue", "-h", "-j", jobid, "-t", "R,CG,CF"], capture_output=True, text=True, timeout=60)
        return bool(o.stdout.strip())
    except Exception:
        return True   # cannot tell: do not compete with it


def claim(path) -> bool:
    path = Path(path)
    if complete(path):
        print(f"run guard: {path.name} is already complete; nothing to do", flush=True)
        return False
    me = os.environ.get("SLURM_JOB_ID", "")
    c = path.with_name(path.name + ".claim")
    if c.exists():
        other = c.read_text().strip()
        if other and other != me and _running(other):
            print(f"run guard: {path.name} is being computed by running job {other}; exiting", flush=True)
            return False
    if me:
        c.parent.mkdir(parents=True, exist_ok=True)
        c.write_text(me + "\n")
    return True
