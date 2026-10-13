"""Gain from the correct geometry vs the model's own competence, across 7 domains (F25; 41 points).

Each point = one (domain, OLMo-2-7B checkpoint): x = held-out test accuracy of the 'none' arm (no geometry loss),
y = test accuracy of the domain's correct geometry minus none (percentage points), mean over the seeds present
(3 seeds; LoRA, lam 20). Domains, arms and result directories (repo results/):
  digits +       helix   081 (2k), 082b (5k), 083 (20k); 10k / 50k / 150k: none from 064 / 066 / 068, helix from 071 / 072 / 073
  digits -       helix   122-126 (5k-150k)
  number words   helix   127-132        Roman   helix   133-138
  days           circle  139-144        months  circle  145-150        letters  line  151-156
Writes the points to report/figures/gain_vs_competence.json and redraws report/figures/gain_vs_competence.png
(binned mean in black: bins < 25, 25-50, 50-75, 75-90, >= 90 % of the none accuracy).

    python tools/gain_vs_competence.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
CK = ("2k", "5k", "10k", "20k", "50k", "150k")
DOMAINS = {   # name: (correct geometry, {ckpt: (dir for none, dir for the geometry)})
    "digits +": ("helix", {"2k": ("081", "081"), "5k": ("082b", "082b"), "10k": ("064", "071"), "20k": ("083", "083"),
                           "50k": ("066", "072"), "150k": ("068", "073")}),
    "digits −": ("helix", {c: (t, t) for c, t in zip(CK[1:], ("122", "123", "124", "125", "126"))}),
    "number words": ("helix", {c: (t, t) for c, t in zip(CK, ("127", "128", "129", "130", "131", "132"))}),
    "Roman": ("helix", {c: (t, t) for c, t in zip(CK, ("133", "134", "135", "136", "137", "138"))}),
    "days": ("circle", {c: (t, t) for c, t in zip(CK, ("139", "140", "141", "142", "143", "144"))}),
    "months": ("circle", {c: (t, t) for c, t in zip(CK, ("145", "146", "147", "148", "149", "150"))}),
    "letters": ("line", {c: (t, t) for c, t in zip(CK, ("151", "152", "153", "154", "155", "156"))}),
}


def mean_test(task: str, geom: str) -> tuple[float, int]:
    vals = []
    for d in glob.glob(str(R / f"{task}-*")):
        for p in glob.glob(f"{d}/run_{geom}_seed*.json"):
            r = json.loads(Path(p).read_text())
            if r.get("geom", geom) == geom and r.get("mode", "lora") == "lora":
                vals.append(r["eval"]["test"])
    if not vals:
        raise FileNotFoundError(f"no {geom} runs in results/{task}-*")
    return float(np.mean(vals)), len(vals)


def collect() -> list[dict]:
    pts = []
    for dom, (geom, cks) in DOMAINS.items():
        for ck, (tn, tg) in cks.items():
            n, nn = mean_test(tn, "none")
            g, ng = mean_test(tg, geom)
            pts.append({"domain": dom, "ckpt": ck, "geometry": geom, "none_acc": 100 * n, "geom_acc": 100 * g,
                        "gain": 100 * (g - n), "seeds_none": nn, "seeds_geom": ng, "tasks": [tn, tg]})
    return pts


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = collect()
    out = REPO / "report/figures"
    (out / "gain_vs_competence.json").write_text(json.dumps(pts, indent=1, ensure_ascii=False))
    bins = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]
    print(f"{len(pts)} points")
    for lo, hi in bins:
        g = [p["gain"] for p in pts if lo <= p["none_acc"] < hi]
        print(f"none {lo}-{hi}%: n={len(g)} mean gain {np.mean(g):+.1f} positive {sum(x > 0 for x in g)}/{len(g)}")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    markers = dict(zip(DOMAINS, "osv^Dx+"))
    for dom in DOMAINS:
        q = sorted((p for p in pts if p["domain"] == dom), key=lambda p: p["none_acc"])
        ax.plot([p["none_acc"] for p in q], [p["gain"] for p in q], marker=markers[dom], linewidth=1, alpha=0.6, label=dom)
    xs = [np.mean([p["none_acc"] for p in pts if lo <= p["none_acc"] < hi]) for lo, hi in bins]
    ys = [np.mean([p["gain"] for p in pts if lo <= p["none_acc"] < hi]) for lo, hi in bins]
    ax.plot(xs, ys, color="black", linewidth=2.5, label="binned mean")
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xlabel("accuracy on unseen problems without the geometry loss (%)")
    ax.set_ylabel("gain from the correct geometry (points)")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "gain_vs_competence.png")


if __name__ == "__main__":
    main()
