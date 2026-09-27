"""Collect exp1/exp2 results across checkpoints into one table and figure (runs anywhere, CPU only).

    python tools/aggregate_sweep.py --results results --out results/sweep_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

# checkpoint label -> (x position for plots, exp1 dir, exp2 dir, grouped-probe dir)
def discover(root: Path) -> list:
    ck = {}
    for d in sorted(root.iterdir()):
        m = re.match(r"\d+-sweep-(.+)$", d.name)
        if m:
            ck.setdefault(m.group(1), {})
            ck[m.group(1)]["exp1"] = d / "exp1"; ck[m.group(1)]["exp2"] = d / "exp2"
        m = re.match(r"\d+-probes-(.+)$", d.name)
        if m:
            ck.setdefault(m.group(1), {})["probes"] = d
    ck.setdefault("main", {})
    if (root / "004-exp1-main").exists():
        ck["main"]["exp1"] = root / "004-exp1-main"
    if (root / "005-exp2-main").exists():
        ck["main"]["exp2"] = root / "005-exp2-main"
    rows = []
    for name, paths in ck.items():
        if name.startswith("s1-"):
            step, stage = int(name[3:]), 1
        elif name.startswith("s2i"):
            step, stage = 928646 + 11931, 2
        else:
            step, stage = 928646 + 2 * 11931, 3   # main = stage-2 soup, drawn after the ingredients
        rows.append({"ckpt": name, "step": step, "stage": stage, **{k: v for k, v in paths.items()}})
    return sorted(rows, key=lambda r: (r["step"], r["ckpt"]))


def exp1_metrics(d: Path) -> dict:
    out = {}
    f = d / "fits.json"
    if not f.exists():
        return out
    fits = json.loads(f.read_text())
    out["acc_2term"] = fits["base_acc"]
    a = [r for r in fits["fits"] if r["slot"] == "A"]
    best = max(a, key=lambda r: r["r2_helix"])
    out["helix_r2_A_max"] = best["r2_helix"]; out["helix_r2_A_layer"] = best["layer"]
    out["wrong_r2_at_that_layer"] = best["r2_wrong_periods"]
    s = d / "steer.csv"
    if s.exists():
        rows = list(csv.DictReader(s.open()))
        def best_layer(cond, deltas, slot="A"):
            by = {}
            for r in rows:
                if r["slot"] == slot and r["cond"] == cond and int(r["delta"]) in deltas:
                    by.setdefault(int(r["layer"]), []).append(float(r["success"]))
            if not by:
                return np.nan
            return max(np.mean(v) for v in by.values())
        small, tens = (-2, -1, 1, 2), (-20, -10, 10, 20)
        out["steer_helix_small"] = best_layer("helix", small)
        out["steer_helix_tens"] = best_layer("helix", tens)
        out["steer_swap_tens"] = best_layer("full_swap", tens)
        out["steer_random_small"] = best_layer("random_orth", small)
    return out


def exp2_metrics(d2: Path | None, dp: Path | None) -> dict:
    out = {}
    if d2 is not None and (d2 / "exp2.json").exists():
        j = json.loads((d2 / "exp2.json").read_text())
        out["acc_3term"] = j["base_acc"]
        ic = [r for r in j.get("interchange", []) if r["donor"] == "diff_sum"]
        if ic:
            mid = [r for r in ic if r["slot"] == "op2" and 4 <= r["layer"] <= 16]
            out["op2_swap_stay_L4_16"] = float(np.mean([r["stay"] for r in mid]))
            a_mid = [r for r in ic if r["slot"] == "A" and 4 <= r["layer"] <= 16]
            out["A_swap_to_a2bc_L4_16"] = float(np.mean([r["to_a_prime_b_c"] for r in a_mid]))
    # grouped probes: the dedicated re-run if present, else the sweep's own exp2 (grouped from 9a94b70 on)
    src = dp if dp is not None and (dp / "probes.csv").exists() else d2
    if src is not None and (src / "probes.csv").exists():
        pr = list(csv.DictReader((src / "probes.csv").open()))
        def mx(slot, t):
            v = [float(r["acc"]) for r in pr if r["slot"] == slot and r["target"] == t]
            return max(v) if v else np.nan
        out["probe_s_B"] = mx("B", "s"); out["probe_s_op2"] = mx("op2", "s"); out["probe_t_eq"] = mx("eq", "t")
        out["probe_s_shuf_op2"] = mx("op2", "s_shuffled")
        out["probe_source"] = src.name
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/sweep_summary")
    args = ap.parse_args()
    root = Path(args.results); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    table = []
    for r in discover(root):
        m = {"ckpt": r["ckpt"], "step": r["step"], "stage": r["stage"]}
        if "exp1" in r:
            m.update(exp1_metrics(r["exp1"]))
        m.update(exp2_metrics(r.get("exp2"), r.get("probes")))
        table.append(m)
    cols = ["ckpt", "acc_2term", "acc_3term", "helix_r2_A_max", "helix_r2_A_layer", "wrong_r2_at_that_layer",
            "steer_helix_small", "steer_helix_tens", "steer_swap_tens", "steer_random_small",
            "probe_s_B", "probe_s_op2", "probe_t_eq", "probe_s_shuf_op2", "op2_swap_stay_L4_16", "A_swap_to_a2bc_L4_16", "probe_source"]
    def fmt(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.2f}"
        return str(v) if v is not None else "-"
    lines = ["| " + " | ".join(cols) + " |\n", "|" + "---|" * len(cols) + "\n"]
    for m in table:
        lines.append("| " + " | ".join(fmt(m.get(c, np.nan)) for c in cols) + " |\n")
    (out / "sweep_table.md").write_text("".join(lines))
    (out / "sweep_table.json").write_text(json.dumps(table, indent=1, default=str))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s1 = [m for m in table if m["stage"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    panels = [
        ("capability", ["acc_2term", "acc_3term"]),
        ("helix: fit and steering (slot A)", ["helix_r2_A_max", "steer_helix_small", "steer_helix_tens", "steer_swap_tens"]),
        ("three-term: running sum", ["probe_s_B", "probe_s_op2", "probe_t_eq", "op2_swap_stay_L4_16"]),
    ]
    for ax, (title, keys) in zip(axes, panels):
        for k in keys:
            xs = [m["step"] for m in s1 if k in m]; ys = [m[k] for m in s1 if k in m]
            line, = ax.plot(xs, ys, "o-", label=k)
            for m in table:
                if m["stage"] > 1 and k in m:
                    ax.plot([m["step"]], [m[k]], "*" if m["ckpt"] == "main" else "s", color=line.get_color(), ms=9)
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05); ax.set_title(title); ax.set_xlabel("stage-1 step (squares: stage-2 ingredients, star: main)")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "sweep.png", dpi=120); plt.close(fig)
    print("".join(lines))


if __name__ == "__main__":
    main()
