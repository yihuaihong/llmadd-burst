"""E5 corpus side: how specifically does each item co-occur with ITS OWN ordinal in OLMo-2's pretraining mix?

For every domain, A[i, j] = exact infini-gram count of the n-grams "<item i> is the <ordinal j>" + "<item i>, the
<ordinal j>" (e.g. "Mars, the fourth"), over the domain's numbers, in the index of OLMo-mix-1124 (the OLMo-2
stage-1 data). (Windowed AND-counts were tried first: their sampled approximation was unusable.)
L = log(1 + A), double-centred (removes item and ordinal frequency); specificity = mean diagonal - mean
off-diagonal, with a column-permutation null (2000), and top1 = share of items whose own ordinal ranks first.
Queries are throttled to 1 per second.

    python tools/corpus_assoc.py --out results/e5-corpus-assoc
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from number_route import DOMAINS

ORD = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
       "fifteenth sixteenth").split()
API = "https://api.infini-gram.io/"


def count(query: str, index: str) -> float:
    body = json.dumps({"index": index, "query_type": "count", "query": query}).encode()
    for attempt in range(8):
        time.sleep(1.0)
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "User-Agent": "llmadd-research/1.0"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            if "error" in r:
                raise RuntimeError(r["error"])
            return float(r["count"])
        except Exception as e:   # noqa: BLE001 - network hiccups: back off and retry
            if attempt == 7:
                raise
            time.sleep(30 * (attempt + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default="v4_olmo-mix-1124_llama")
    ap.add_argument("--domains", default="days,months,planets,elements,presidents,greek,solfege")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = {"index": args.index, "templates": ["{item} is the {ord}", "{item}, the {ord}"], "domains": {}}
    for d in args.domains.split(","):
        D = DOMAINS[d]
        items = D["items"]
        vals = np.array(D.get("values", range(1, len(items) + 1)))
        A = np.array([[count(f"{w} is the {ORD[v - 1]}", args.index) + count(f"{w}, the {ORD[v - 1]}", args.index)
                       for v in vals] for w in items])
        L = np.log1p(A)
        Lc = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
        n = len(items)
        off = ~np.eye(n, dtype=bool)
        spec = lambda M: float(np.diag(M).mean() - M[off].mean())
        obs = spec(Lc)
        rng = np.random.default_rng(0)
        null = np.array([spec(Lc[:, rng.permutation(n)]) for _ in range(2000)])
        res["domains"][d] = {"items": items, "values": vals.tolist(), "counts": A.tolist(), "specificity": obs,
                             "null_mean": float(null.mean()), "p": float(((null >= obs).sum() + 1) / 2001),
                             "top1": float((Lc.argmax(1) == np.arange(n)).mean())}
        print(f"{d}: specificity {obs:.2f} (null {null.mean():.2f}, p {res['domains'][d]['p']:.4f}) top1 {res['domains'][d]['top1']:.2f}", flush=True)
        (out / "assoc.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
