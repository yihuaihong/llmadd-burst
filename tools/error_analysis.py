"""Error anatomy of manifold_train.py runs made with --save_preds.

For every run directory and geometry (pooled over seeds) and the eval sets test / three_digit it reports:
accuracy, share of non-number outputs, and for the WRONG numeric answers: how many have fewer digits than the
truth (range collapse, e.g. a two-digit-sum prior), which digit positions differ (units / tens / hundreds only,
or several), and accuracy split by whether the problem needs a units / tens carry (add) or borrow (sub).

    python tools/error_analysis.py <run_dir> [<run_dir> ...] > error_analysis.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def digits(v: np.ndarray, k: int) -> np.ndarray:
    return (v // 10 ** k) % 10


def carries(P: np.ndarray, task: str) -> tuple:
    a, b = P[:, 0], P[:, 1]
    if task == "add":
        c1 = digits(a, 0) + digits(b, 0) >= 10
        c2 = digits(a, 1) + digits(b, 1) + c1 >= 10
    else:
        c1 = digits(a, 0) < digits(b, 0)
        c2 = digits(a, 1) - c1 < digits(b, 1)
    return c1, c2


def anatomy(P: np.ndarray, pred: np.ndarray, task: str) -> dict:
    true = P[:, 0] + P[:, 1] if task == "add" else P[:, 0] - P[:, 1]
    ok = pred == true
    wrong = ~ok & (pred >= 0)
    nd = lambda v: np.where(v > 0, np.floor(np.log10(np.maximum(v, 1))) + 1, 1)
    fewer = wrong & (nd(pred) < nd(true))
    same = wrong & (nd(pred) == nd(true))
    diff = np.stack([digits(pred, k) != digits(true, k) for k in range(3)], 1)
    nw = max(int(wrong.sum()), 1)
    c1, c2 = carries(P, task)
    acc_if = lambda m: float(ok[m].mean()) if m.any() else float("nan")
    return {
        "n": len(P), "acc": float(ok.mean()), "non_number": float((pred < 0).mean()),
        "wrong_fewer_digits": float(fewer.sum() / nw),
        "wrong_units_only": float((same & diff[:, 0] & ~diff[:, 1] & ~diff[:, 2]).sum() / nw),
        "wrong_tens_only": float((same & ~diff[:, 0] & diff[:, 1] & ~diff[:, 2]).sum() / nw),
        "wrong_hundreds_only": float((same & ~diff[:, 0] & ~diff[:, 1] & diff[:, 2]).sum() / nw),
        "wrong_several": float((same & (diff.sum(1) > 1)).sum() / nw),
        "acc_no_carry": acc_if(~c1 & ~c2), "acc_units_carry": acc_if(c1), "acc_tens_carry": acc_if(c2),
    }


def main() -> None:
    cols = ["acc", "non_number", "wrong_fewer_digits", "wrong_units_only", "wrong_tens_only", "wrong_hundreds_only",
            "wrong_several", "acc_no_carry", "acc_units_carry", "acc_tens_carry"]
    out = ["# error anatomy (pooled over seeds; the wrong_* shares are fractions of the wrong numeric answers)\n"]
    for d in map(Path, sys.argv[1:]):
        base = json.loads((d / "base.json").read_text())
        task = base.get("task", "add")
        items = {k: np.array(v) for k, v in base["items"].items()}
        runs = [json.loads(p.read_text()) for p in sorted(d.glob("run_*.json"))]
        runs = [r for r in runs if r.get("preds")]
        for es in ("three_digit", "test"):
            out.append(f"\n## {d.name}: {es} (task {task}, split {base.get('split', 'random')})\n\n")
            out.append("| geom | seeds | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols) + "\n")
            rows = [("base", [base["preds"]] if base.get("preds") else [])]
            for G in dict.fromkeys(r["geom"] for r in runs):
                rows.append((G, [r["preds"] for r in runs if r["geom"] == G]))
            for G, preds in rows:
                if not preds:
                    continue
                P = np.concatenate([items[es]] * len(preds))
                pr = np.concatenate([np.array(p[es]) for p in preds])
                a = anatomy(P, pr, task)
                out.append(f"| {G} | {len(preds)} | " + " | ".join(f"{a[c]:.3f}" for c in cols) + " |\n")
    print("".join(out))


if __name__ == "__main__":
    main()
