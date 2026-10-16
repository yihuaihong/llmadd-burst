"""Summary of the 36-domain fine-tuning prior screen (runs/ft36; tools/ft36.py; intent spec runs/ft36/AUDIT.md).

Reads <out>/<ckpt>/<domain>/{base,run_none,run_prior,run_prior_shuf}_seed<s>.json, seeds 0-2. One cell = checkpoint x
domain (2 x 36 = 72 cells). Per cell:
  acc     held-out test accuracy x100 of base (adapter disabled) / none / prior / prior_shuf: mean and per seed; n_test per
          seed (the split follows the seed and is the same for every arm of that seed); chance x100; train accuracy x100
          (<= 400 training questions, the same ones for every arm of a seed)
  band    competence of none (mean over seeds): floor <= chance + 5 points, ceiling >= 95, middle otherwise
  P-N     prior - none (points), seed wins, paired t over the 3 seeds (df 2, reference only). Item level: the per-question
          test hits of the two arms are paired WITHIN a seed (same questions, same order); seeds are strata and the
          discordant counts are summed over them (b = prior right & none wrong, c = the reverse): exact McNemar (binomial)
          on b vs c, i.e. the stratified (CMH) McNemar for matched pairs. Holm over the 72 cells. P-S = prior - prior_shuf
          and S-N = prior_shuf - none in the same way, each its own Holm family of 72.
          The item test is conditional on the trained adapters (training-seed variance is not in it): read it with the
          seed wins.
  clus    robustness of the item test. Questions of one split unit (the key ft36 holds out: an unordered option pair
          {b, c} with both orders and every anchor; a cyclic class (item, k mod n); an offset key) are held out together
          and share items, and a key held out in several seeds enters once per seed: the splits are rebuilt from the
          snapshot code (ft36.offset_questions / pair_questions + the make_split rule of ft36.main, checked against
          n_train / n_test / n_test_answer_trained / chance of every JSON), and each key's paired difference summed over
          its questions and seeds is sign-flipped as one unit (20000 flips, two-sided; Holm over 72). Needs the env of the
          runs (torch, geonamescache); --no-recon skips it (everything else uses the JSONs only).
  robust  Holm p < .05 in BOTH the item test and the key-cluster test, the per-seed differences all of one sign, and
          leave-one-seed-out: the item test on each pair of seeds alone has p < .05 in the same direction (an effect
          carried by one adapter fails; ft36 results audit 2026-10-16).
Information (audit I5; needs recon): for comparison domains, the share of held-out answers implied by the TARGET (anchor
tasks: the option nearer the anchor in the target space Fz as ft36 builds it; no anchor: sign of a linear readout of
Fz[b] - Fz[c] fitted on the training questions) and by the TRAINING questions alone (transitive closure: one graph per
anchor, edge nearer -> farther, or one graph winner -> loser), and P-N on the held-out questions the training does not
imply (there a gain can only come from the target or from pretraining knowledge). For offset domains: whether the
training questions connect all items (then the item order is implied by training and the target adds no answer) and
the share of held-out questions whose (item, answer) pair never occurs in training (by construction 100%: a model that
learns per-item answer frequencies scores below chance, F80 M2).
Pooling: families by atlas kind (numeric helix / cyclic / ordinal-line / spatial sphere-plane / tree / comparison-scalar;
emotions = plane kind, valence task), also split by task type (offset "k after x", chance 1/n; comparison, chance 1/2).
Gain vs competence (F25 / F26 inverted U): x = chance-normalised none accuracy (none - chance) / (1 - chance), y = the
chance-normalised gain (prior - none) / (1 - chance); the F25 bins of x, Spearman(x, y), Spearman(x(1 - x), y) (an inverted
U centred at 0.5 makes it positive), the quadratic term of y ~ x + x^2 (95% CI from resampling domains, both checkpoints of
a domain together), and Spearman on each side of x = 0.5; separately for offset cells (target carries no answers), offset
cells above the floor, and comparison cells (target carries the answers), because competence and task type are confounded.
Domain lists for the next design (ft36b / pretraining manifolds): measurable = none above floor and <= 90 (headroom >= 10).
Integrity: 4 files x 3 seeds per cell and nothing else in the directory, readable, arm / seed / split_seed / steps / lam /
layers / geo_bs as launched, prior_skipped == 0, len(test_hits) == n_test and mean(test_hits) == test, the info fields equal
over the 4 files of a seed, the base equal across seeds where it must be (probe geometry, which does not depend on the
split; hits on questions held out in several seeds, recon) - each base was measured right after a different trained
adapter (audit B1), the prior acted (CKA of the held-out item states with the TRUE target vs base, for prior up and for
prior_shuf down; the logged training geo loss 1 - CKA against the arm's OWN target, the shuffled one for prior_shuf, from
step 0 to step 210), train accuracy of none (ceiling control), numbers @ 42B positive control.

    /scratch/yh6210/miniconda3/envs/open_r1/bin/python tools/ft36_summary.py /scratch/yh6210/llmadd_torch/runs/ft36/out
        [--no-recon] [--code <snapshot tools dir; default <out>/../code/tools>] [--json <cells.json>]
"""
import argparse
import collections
import itertools
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", message="Precision loss")   # ttest_rel on near-equal seed differences

ARMS = ("none", "prior", "prior_shuf")
FILES = ("base",) + ARMS
SEEDS = (0, 1, 2)
LAUNCH = {"steps": 280, "lam": 20.0, "geo_bs": 64}
LAYERS = [4, 8, 12, 16]
INFO = ("domain", "primary", "raw_gram", "split_seed", "n_train", "n_test", "n_test_answer_trained", "chance", "layers", "model")
FAMILY = {
    "numeric helix": "numbers number_words ordinals years",
    "cyclic": "days months hours pitch keys zodiac chinese_zodiac colours compass",
    "ordinal-line": "letters greek planets elements geo_periods presidents ranks life_stages prefixes people",
    "spatial sphere-plane": "stars cities countries landmarks us_cities emotions",
    "tree": "animals languages",
    "comparison-scalar": "country_size animal_size kinship scalar_adj units",
}
FAM = {d: f for f, ds in FAMILY.items() for d in ds.split()}
assert len(FAM) == 36
OFFSET_DOMAINS = set("numbers number_words ordinals years days months hours letters greek planets elements pitch keys "
                     "geo_periods zodiac chinese_zodiac".split())
ANCHORED = set("stars cities countries landmarks us_cities colours compass animals languages".split())   # "closer to {a}"
INFO_CLASS = {d: "offset" if d in OFFSET_DOMAINS else "cmp-anchor" if d in ANCHORED else "cmp-order" for d in FAM}
BINS = ((-1e9, 25), (25, 50), (50, 75), (75, 90), (90, 1e9))   # F25 bins of none accuracy (here chance-normalised, x100)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--no-recon", action="store_true")
ap.add_argument("--code", default=None)
ap.add_argument("--json", default=None)
ap.add_argument("--flips", type=int, default=20000)
args = ap.parse_args()
ROOT = Path(args.out)
CODE = Path(args.code) if args.code else ROOT.parent / "code" / "tools"


def tokens(ck):
    return float(re.match(r"([0-9.]+)B", ck).group(1))


def fname(a, s):
    return f"base_seed{s}.json" if a == "base" else f"run_{a}_seed{s}.json"


# ------------------------------------------------------------------------------------------------------------ load
problems = []
cells = {}
for ckd in sorted((p for p in ROOT.iterdir() if p.is_dir()), key=lambda p: tokens(p.name)):
    for dd in sorted(p for p in ckd.iterdir() if p.is_dir()):
        key = (ckd.name, dd.name)
        want = {fname(a, s) for a in FILES for s in SEEDS}
        extra = sorted({f.name for f in dd.iterdir()} - want)
        if extra:
            problems.append(f"{key}: unexpected files {extra}")
        c = {}
        for a in FILES:
            for s in SEEDS:
                try:
                    r = json.loads((dd / fname(a, s)).read_text())
                    assert "test" in r
                    c[a, s] = r
                except Exception as e:
                    problems.append(f"{key}: {fname(a, s)} missing or unreadable ({type(e).__name__})")
        cells[key] = c
CKS = sorted({k[0] for k in cells}, key=tokens)
DOMS = sorted({k[1] for k in cells})
complete = {k for k, c in cells.items() if len(c) == len(FILES) * len(SEEDS)}

# ------------------------------------------------------------------------------------------------------- integrity
for key in sorted(complete):
    ck, dom = key
    c = cells[key]
    if dom not in FAM:
        problems.append(f"{key}: domain not in FAMILY")
    for s in SEEDS:
        b = c["base", s]
        for a in FILES:
            r = c[a, s]
            for f in INFO:
                if r.get(f) != b.get(f):
                    problems.append(f"{key} seed {s}: {a} {f} = {r.get(f)} != base {b.get(f)}")
            if r["split_seed"] != s:
                problems.append(f"{key} {a} seed {s}: split_seed {r['split_seed']}")
            if r["layers"] != LAYERS:
                problems.append(f"{key} {a} seed {s}: layers {r['layers']}")
            if not r["model"].rstrip("/").endswith(f"tokens{ck}"):
                problems.append(f"{key} {a} seed {s}: model {r['model']}")
            if r["domain"] != dom:
                problems.append(f"{key} {a} seed {s}: domain {r['domain']}")
            h = r["test_hits"]
            if len(h) != r["n_test"] or abs(np.mean(h) - r["test"]) > 1e-9 or set(h) - {0, 1}:
                problems.append(f"{key} {a} seed {s}: test_hits inconsistent with n_test / test")
            if a == "base":
                continue
            if r["arm"] != a or r["seed"] != s:
                problems.append(f"{key} {fname(a, s)}: arm / seed fields {r['arm']} / {r['seed']}")
            for f, v in LAUNCH.items():
                if r[f] != v:
                    problems.append(f"{key} {a} seed {s}: {f} = {r[f]}")
            if r["prior_skipped"] != 0:
                problems.append(f"{key} {a} seed {s}: prior_skipped = {r['prior_skipped']}")
            if (r["shuf_cka"] is None) != (a != "prior_shuf"):
                problems.append(f"{key} {a} seed {s}: shuf_cka = {r['shuf_cka']}")
            steps = [e["step"] for e in r["log"]]
            if steps != [0, 70, 140, 210]:
                problems.append(f"{key} {a} seed {s}: log steps {steps}")
            if (a == "none") != all(e["geo"] is None for e in r["log"]):
                problems.append(f"{key} {a} seed {s}: geo log {[e['geo'] for e in r['log']]}")
    for f in ("primary", "raw_gram", "model"):
        if len({c["base", s][f] for s in SEEDS}) != 1:
            problems.append(f"{key}: {f} differs across seeds")
    pr = np.array([[c["base", s]["geometry"]["probe"][f"L{L}"] for L in LAYERS] for s in SEEDS])
    if np.abs(pr - pr[0]).max() > 1e-4:
        problems.append(f"{key}: base probe CKA differs across seeds by {np.abs(pr - pr[0]).max():.2e} (adapter leak?)")

# ------------------------------------------------------------------------------- split reconstruction (optional)
recon, recon_note = {}, "skipped (--no-recon)"
if not args.no_recon:
    try:
        sys.path.insert(0, str(CODE))
        import ft36
        from manifold_domains import OFFSET, domains
        D = domains()

        def make_split(qs, cyclic, sseed):   # = make_split inside ft36.main (snapshot 8be3a01), checked below
            keys = [q["key"] for q in qs]
            held = ft36.split(keys, 0.4, sseed)
            if cyclic:
                held = {h for h in held if h[1] != 0}
            elif cyclic is False:
                by_key = {}
                for q in qs:
                    by_key.setdefault(q["key"], []).append(q)
                while True:
                    tr = [q for q in qs if q["key"] not in held]
                    xs, ks_ = {q["xk"][0] for q in tr}, {q["xk"][1] for q in tr}
                    bad = [h for h in sorted(held) if any(q["xk"][0] not in xs or q["xk"][1] not in ks_ for q in by_key[h])]
                    if not bad:
                        break
                    held.discard(bad[0])
            return [q for q in qs if q["key"] not in held], [q for q in qs if q["key"] in held]

        from manifold_atlas import primary
        from manifold_domains import PAIR, PAIR2

        def reach(adj, s0, t0):
            seen, st = {s0}, [s0]
            while st:
                u = st.pop()
                for v in adj.get(u, ()):
                    if v == t0:
                        return True
                    if v not in seen:
                        seen.add(v)
                        st.append(v)
            return False

        def components(edges, nodes):
            par = {v: v for v in nodes}

            def find(v):
                while par[v] != v:
                    par[v] = par[par[v]]
                    v = par[v]
                return v
            for u, v in edges:
                par[find(u)] = find(v)
            return len({find(v) for v in nodes})

        for dom in DOMS:
            d = D[dom]
            assert (dom in OFFSET) == (dom in OFFSET_DOMAINS), dom
            qs, cyclic = ft36.offset_questions(dom, d) if dom in OFFSET else (ft36.pair_questions(dom, d), None)
            n_items = len(d["items"])
            right = (lambda q: q["ans"]) if dom in OFFSET else (lambda q: q["opts"][q["ans"]])
            tn = primary(d, dom)
            assert tn == cells[next(k for k in complete if k[1] == dom)]["base", 0]["primary"], dom
            T = np.asarray(d["targets"][tn], dtype=np.float64)
            T = T[:, T.std(0) > 1e-9]
            Fz = (T - T.mean(0)) if tn in ft36.RAW_GRAM else (T - T.mean(0)) / T.std(0)   # = ft36.main
            anchored = dom not in OFFSET and "{a}" in (PAIR2[dom][0] if dom in PAIR2 else PAIR[dom])
            assert anchored == (dom in ANCHORED), dom
            for s in SEEDS:
                train, test = make_split(qs, cyclic, s)
                ids = [q["prompt"] for q in test]
                assert len(set(ids)) == len(ids), dom
                ans_tr = {right(q) for q in train}
                recon[dom, s] = {"ids": ids, "keys": [q["key"] for q in test], "n_train": len(train), "n_test": len(test),
                                 "n_test_answer_trained": int(sum(right(q) in ans_tr for q in test)),
                                 "chance": float(np.mean([1 / (n_items if dom in OFFSET else 2) for _ in test]))}
                if dom in OFFSET:
                    tr_pairs = {(q["occ"][0][0], q["ans"]) for q in train}
                    nodes = {q["occ"][0][0] for q in qs} | {q["ans"] for q in qs}
                    recon[dom, s].update(
                        anchored=None,
                        new_pair=float(np.mean([(q["occ"][0][0], q["ans"]) not in tr_pairs for q in test])),
                        n_comp=components([(q["occ"][0][0], q["ans"]) for q in train], nodes))
                    continue
                anc = (lambda q: q["occ"][2][0]) if anchored else (lambda q: -1)
                if anchored:
                    def tpred(q):
                        a, (x, y) = anc(q), q["opts"]
                        return 0 if np.sum((Fz[a] - Fz[x]) ** 2) < np.sum((Fz[a] - Fz[y]) ** 2) else 1
                else:
                    w = np.linalg.lstsq(np.array([Fz[q["opts"][0]] - Fz[q["opts"][1]] for q in train]),
                                        np.array([1.0 if q["ans"] == 0 else -1.0 for q in train]), rcond=None)[0]

                    def tpred(q):
                        return 0 if (Fz[q["opts"][0]] - Fz[q["opts"][1]]) @ w > 0 else 1
                adj = {}
                for q in train:
                    adj.setdefault((anc(q), q["opts"][q["ans"]]), set()).add((anc(q), q["opts"][1 - q["ans"]]))
                recon[dom, s].update(
                    anchored=anchored,
                    target=np.array([tpred(q) == q["ans"] for q in test]),
                    closure=np.array([reach(adj, (anc(q), q["opts"][q["ans"]]), (anc(q), q["opts"][1 - q["ans"]])) for q in test]))
        bad = 0
        for key in sorted(complete):
            for s in SEEDS:
                r, q = cells[key]["base", s], recon[key[1], s]
                for f in ("n_train", "n_test", "n_test_answer_trained"):
                    if r[f] != q[f]:
                        bad += 1
                        problems.append(f"{key} seed {s}: rebuilt {f} {q[f]} != JSON {r[f]}")
                if abs(r["chance"] - q["chance"]) > 1e-12:
                    bad += 1
                    problems.append(f"{key} seed {s}: rebuilt chance {q['chance']} != JSON {r['chance']}")
        recon_note = (f"rebuilt from {CODE}: {len(recon)} domain x seed splits; n_train / n_test / n_test_answer_trained / "
                      f"chance {'all equal to the JSONs' if not bad else f'{bad} MISMATCHES'}")
    except Exception as e:   # the summary itself only needs the JSONs
        recon, recon_note = {}, f"FAILED ({type(e).__name__}: {e}); cluster test and cross-seed base hits not run"

base_shared = collections.Counter()
if recon:
    for key in sorted(complete):
        for s, t in itertools.combinations(SEEDS, 2):
            hs = dict(zip(recon[key[1], s]["ids"], cells[key]["base", s]["test_hits"]))
            ht = dict(zip(recon[key[1], t]["ids"], cells[key]["base", t]["test_hits"]))
            sh = set(hs) & set(ht)
            base_shared["shared"] += len(sh)
            dis = sum(hs[u] != ht[u] for u in sh)
            base_shared["differ"] += dis
            if dis:
                base_shared[key] += dis

# ----------------------------------------------------------------------------------------------------- statistics
RNG = np.random.default_rng(0)


def mcnemar(key, a1, a2):
    """Discordant counts per seed (a1 right & a2 wrong, a1 wrong & a2 right) and the exact McNemar p on their sums."""
    per = []
    for s in SEEDS:
        x, y = np.array(cells[key][a1, s]["test_hits"]), np.array(cells[key][a2, s]["test_hits"])
        per.append((int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())))
    b, c = sum(p[0] for p in per), sum(p[1] for p in per)
    return per, b, c, (stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def cluster_flip(key, a1, a2):
    if not recon:
        return np.nan
    d = collections.defaultdict(int)
    for s in SEEDS:   # one unit = one split key (all its questions, all seeds in which it is held out)
        for u, x, y in zip(recon[key[1], s]["keys"], cells[key][a1, s]["test_hits"], cells[key][a2, s]["test_hits"]):
            d[u] += x - y
    v = np.array([x for x in d.values() if x != 0], dtype=np.float64)
    if not len(v):
        return 1.0
    T = abs(v.sum())
    if len(v) <= 16:   # exact
        sg = np.array(list(itertools.product((-1, 1), repeat=len(v))))
        return float(np.mean(np.abs(sg @ v) >= T - 1e-9))
    hit = 0
    for _ in range(args.flips // 4000):
        sg = RNG.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(4000, len(v)))
        hit += int((np.abs(sg @ v.astype(np.float32)) >= T - 1e-6).sum())
    return (1 + hit) / (1 + 4000 * (args.flips // 4000))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    out, run, m = np.empty(len(ps)), 0.0, len(ps)
    for rank, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def ttest(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return stats.ttest_rel(x, y).pvalue if np.std(x - y) > 0 else np.nan


COMP = (("P-N", "prior", "none"), ("P-S", "prior", "prior_shuf"), ("S-N", "prior_shuf", "none"))
res = {}
for key in sorted(complete, key=lambda k: (tokens(k[0]), k[1])):
    c = cells[key]
    acc = {a: [100 * c[a, s]["test"] for s in SEEDS] for a in FILES}
    tr = {a: [100 * c[a, s]["train"] for s in SEEDS] for a in FILES}
    chance = 100 * np.mean([c["base", s]["chance"] for s in SEEDS])
    none = np.mean(acc["none"])
    band = "floor" if none <= chance + 5 else "ceiling" if none >= 95 else "middle"
    ctx = {a: [np.mean([c[a, s]["geometry"]["ctx"][f"L{L}"] for L in LAYERS]) for s in SEEDS] for a in FILES}
    probe = {a: [np.mean([c[a, s]["geometry"]["probe"][f"L{L}"] for L in LAYERS]) for s in SEEDS] for a in FILES}
    geo = {a: [(c[a, s]["log"][0]["geo"], c[a, s]["log"][-1]["geo"]) for s in SEEDS] for a in ("prior", "prior_shuf")}
    r = {"ck": key[0], "domain": key[1], "family": FAM.get(key[1], "?"), "task": "off" if key[1] in OFFSET_DOMAINS else "cmp",
         "primary": c["base", 0]["primary"], "n_test": [c["base", s]["n_test"] for s in SEEDS],
         "n_train": [c["base", s]["n_train"] for s in SEEDS],
         "ans_tr": [c["base", s]["n_test_answer_trained"] / c["base", s]["n_test"] for s in SEEDS],
         "chance": chance, "acc": acc, "mean": {a: float(np.mean(v)) for a, v in acc.items()},
         "train": {a: float(np.mean(v)) for a, v in tr.items()}, "train_seeds": tr, "band": band,
         "x": (none - chance) / (100 - chance) * 100, "ctx": ctx, "probe": probe, "geo": geo,
         "shuf_cka": [c["prior_shuf", s]["shuf_cka"] for s in SEEDS]}
    for name, a1, a2 in COMP:
        dd = [x - y for x, y in zip(acc[a1], acc[a2])]
        per, b, cc, p = mcnemar(key, a1, a2)
        loso = []   # leave one seed out: (b, c, p) of the other two seeds
        for s in SEEDS:
            b2, c2 = sum(per[t][0] for t in SEEDS if t != s), sum(per[t][1] for t in SEEDS if t != s)
            loso.append((b2, c2, stats.binomtest(b2, b2 + c2).pvalue if b2 + c2 else 1.0))
        r[name] = {"d": float(np.mean(dd)), "seeds": dd, "wins": int(sum(x > 1e-9 for x in dd)), "losses": int(sum(x < -1e-9 for x in dd)),
                   "t_p": ttest(acc[a1], acc[a2]), "per": per, "b": b, "c": cc, "p": p, "p_clus": cluster_flip(key, a1, a2),
                   "loso": loso}
    r["y"] = r["P-N"]["d"] / (100 - chance) * 100
    res[key] = r
KEYS = list(res)
for name, _, _ in COMP:
    for k, h in zip(KEYS, holm([res[k][name]["p"] for k in KEYS])):
        res[k][name]["holm"] = h
    if recon:
        for k, h in zip(KEYS, holm([res[k][name]["p_clus"] for k in KEYS])):
            res[k][name]["holm_clus"] = h
    for k in KEYS:
        t = res[k][name]
        up = t["b"] > t["c"]
        why = []
        if t["holm"] >= 0.05:
            why.append("item Holm")
        if t.get("holm_clus", 0.0) >= 0.05:
            why.append(f"key-cluster Holm {t['holm_clus']:.2g}")
        if (t["wins"] if up else t["losses"]) < 3:
            why.append(f"seeds {t['wins'] if up else t['losses']}/3")
        bad = [f"s{s} out {b2}:{c2} p {p2:.2g}" for s, (b2, c2, p2) in zip(SEEDS, t["loso"]) if p2 >= 0.05 or (b2 > c2) != up]
        if bad:
            why.append("leave-one-seed-out " + ", ".join(bad))
        t["robust"], t["why"] = not why, why


def sig(t):
    """'+' / '-' when Holm p < 0.05 (item test), by the sign of the effect."""
    return ("+" if t["b"] > t["c"] else "-") if t["holm"] < 0.05 else " "


def f1(v):
    return f"{v:5.1f}"


# ------------------------------------------------------------------------------------------------------------ print
print(f"== integrity ({len(cells)} cells, {len(complete)} complete; split reconstruction {recon_note})")
for p in problems:
    print("  PROBLEM", p)
if not problems:
    print("  no problems: every cell has 4 files x 3 seeds and nothing else, fields as launched (steps 280, lam 20, geo_bs 64,"
          " layers 4/8/12/16), prior_skipped == 0 everywhere, test_hits consistent, info fields equal over the 4 files of a"
          " seed, base probe CKA identical across seeds")
if recon:
    print(f"  base hits on questions held out in 2 seeds: {base_shared['shared']} shared question-pairs, {base_shared['differ']} differ"
          + (" in " + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in base_shared.items() if isinstance(k, tuple)) if base_shared["differ"] else ""))
skip = [k for k in cells if k not in complete]
if skip:
    print("  left out (incomplete):", skip)

for ck in CKS:
    rows = [res[k] for k in KEYS if k[0] == ck]
    rows.sort(key=lambda r: (list(FAMILY).index(r["family"]), r["task"], r["domain"]))
    print(f"\n== {ck}: test accuracy x100 (mean | seeds 0/1/2), n_test per seed, chance; train x100 base/none/prior/shuf; band of none")
    print(f"  {'domain':15s} {'family':12s} {'task':4s} {'target':10s} {'n_test':13s} {'ans_tr':>6s} {'chance':>6s} "
          f"{'base':>5s} | {'none':>5s} {'(seeds)':17s} | {'prior':>5s} {'(seeds)':17s} | {'shuf':>5s} {'(seeds)':17s} | train b/n/p/s       band")
    for r in rows:
        ps = lambda a: "/".join(f"{v:5.1f}" for v in r["acc"][a])
        print(f"  {r['domain']:15s} {r['family'][:12]:12s} {r['task']:4s} {r['primary']:10s} {'/'.join(map(str, r['n_test'])):13s} "
              f"{100 * np.mean(r['ans_tr']):5.1f}% {r['chance']:6.1f} {r['mean']['base']:5.1f} | {r['mean']['none']:5.1f} {ps('none')} | "
              f"{r['mean']['prior']:5.1f} {ps('prior')} | {r['mean']['prior_shuf']:5.1f} {ps('prior_shuf')} | "
              f"{r['train']['base']:3.0f}/{r['train']['none']:3.0f}/{r['train']['prior']:3.0f}/{r['train']['prior_shuf']:3.0f}  {r['band']}")
    print(f"\n== {ck}: paired comparisons. d = mean difference (points), w = seed wins/losses, t = paired t over seeds (df 2); item = "
          f"discordant b:c per seed, exact stratified McNemar p, Holm-72 p (+/- = Holm p < .05); clus = split-key-cluster sign-flip p, R = robust (see docstring)")
    print(f"  {'domain':15s} {'P-N d':>6s} {'w':>4s} {'t':>5s}  {'item b:c (s0 s1 s2)':24s} {'p':>8s} {'holm':>8s} {'clus':>8s} | "
          f"{'P-S d':>6s} {'w':>4s} {'p':>8s} {'holm':>8s} | {'S-N d':>6s} {'w':>4s} {'p':>8s} {'holm':>8s}")
    for r in rows:
        t, u, v = r["P-N"], r["P-S"], r["S-N"]
        bc = " ".join(f"{b}:{c}" for b, c in t["per"])
        print(f"  {r['domain']:15s} {t['d']:+6.1f} {t['wins']}/{t['losses']} {t['t_p']:5.3f}  {bc:24s} {t['p']:8.2g} {t['holm']:8.2g}{sig(t)}"
              f"{t['p_clus']:8.2g}{'R' if t['robust'] else ' '}| {u['d']:+6.1f} {u['wins']}/{u['losses']} {u['p']:8.2g} {u['holm']:8.2g}{sig(u)}| "
              f"{v['d']:+6.1f} {v['wins']}/{v['losses']} {v['p']:8.2g} {v['holm']:8.2g}{sig(v)}")

print("\n== Holm-significant item tests (72 cells each): +/- = direction")
for name, _, _ in COMP:
    pos = [f"{k[0]}/{k[1]} {res[k][name]['d']:+.1f}" for k in KEYS if res[k][name]["holm"] < 0.05 and res[k][name]["b"] > res[k][name]["c"]]
    neg = [f"{k[0]}/{k[1]} {res[k][name]['d']:+.1f}" for k in KEYS if res[k][name]["holm"] < 0.05 and res[k][name]["b"] < res[k][name]["c"]]
    print(f"  {name}: {len(pos)} positive: {', '.join(pos)}")
    print(f"  {name}: {len(neg)} negative: {', '.join(neg)}")
    for sgn, lab in ((1, "positive"), (-1, "negative")):
        rob = [f"{k[0]}/{k[1]} {res[k][name]['d']:+.1f}" for k in KEYS if res[k][name]["robust"]
               and np.sign(res[k][name]["b"] - res[k][name]["c"]) == sgn]
        print(f"  {name}: {len(rob)} ROBUST {lab}{' (no key-cluster test: --no-recon)' if not recon else ''}: {', '.join(rob)}")
    nr = [f"{k[0]}/{k[1]} {res[k][name]['d']:+.1f} [{'; '.join(res[k][name]['why'])}]" for k in KEYS
          if res[k][name]["holm"] < 0.05 and not res[k][name]["robust"]]
    print(f"  {name}: item-Holm-significant but not robust: {', '.join(nr) if nr else 'none'}")


def pool(rows):
    """Equal-weight domain mean of P-N (raw and chance-normalised), domain counts, Holm counts, pooled item test."""
    d = np.array([r["P-N"]["d"] for r in rows])
    pos, neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    b, c = sum(r["P-N"]["b"] for r in rows), sum(r["P-N"]["c"] for r in rows)
    return (f"{len(rows):2d} cells  none {np.mean([r['mean']['none'] for r in rows]):5.1f}  P-N {d.mean():+6.1f} (norm "
            f"{np.mean([r['y'] for r in rows]):+6.1f})  domains +{pos}/-{neg}/={len(rows) - pos - neg} (sign p "
            f"{stats.binomtest(pos, pos + neg).pvalue if pos + neg else 1:.3f})  Holm +{sum(r['P-N']['holm'] < .05 and r['P-N']['b'] > r['P-N']['c'] for r in rows)}"
            f"/-{sum(r['P-N']['holm'] < .05 and r['P-N']['b'] < r['P-N']['c'] for r in rows)}  robust "
            f"+{sum(r['P-N']['robust'] and r['P-N']['b'] > r['P-N']['c'] for r in rows)}"
            f"/-{sum(r['P-N']['robust'] and r['P-N']['b'] < r['P-N']['c'] for r in rows)}  pooled items {b}:{c} "
            f"(p {stats.binomtest(b, b + c).pvalue if b + c else 1:.2g})  P-S {np.mean([r['P-S']['d'] for r in rows]):+6.1f}  "
            f"S-N {np.mean([r['S-N']['d'] for r in rows]):+6.1f}")


INFO_ROWS = {}
if recon:
    print("\n== information the target carries beyond the training questions (audit I5)")
    off = sorted(d for d in DOMS if INFO_CLASS[d] == "offset")
    conn = [d for d in off if all(recon[d, s]["n_comp"] == 1 for s in SEEDS)]
    newp = min(recon[d, s]["new_pair"] for d in off for s in SEEDS)
    print(f"  offset ({len(off)} domains; the target gives item positions only, the prior acts on the item x, never on k or the "
          f"answer): training questions connect all items (item order implied by training) in every seed: {len(conn)}/{len(off)}"
          f"{'' if len(conn) == len(off) else ' NOT: ' + ', '.join(sorted(set(off) - set(conn)))}; held-out questions whose "
          f"(item, answer) pair never occurs in training: min over domains x seeds {100 * newp:.1f}%")
    print("  comparison: target = % of held-out answers implied by the target, closure = % implied by the training questions"
          " (transitive); on the held-out questions NOT implied by training (pooled seeds): n, none / prior / shuf accuracy,"
          " P-N b:c, McNemar p")
    for dom in sorted(d for d in DOMS if INFO_CLASS[d] != "offset"):
        tg = np.mean([recon[dom, s]["target"].mean() for s in SEEDS])
        cl = np.mean([recon[dom, s]["closure"].mean() for s in SEEDS])
        parts = []
        for ck in CKS:
            if (ck, dom) not in res:
                continue
            h = {a: np.concatenate([np.array(cells[ck, dom][a, s]["test_hits"])[~recon[dom, s]["closure"]] for s in SEEDS])
                 for a in ARMS}
            b = int(((h["prior"] == 1) & (h["none"] == 0)).sum())
            c = int(((h["prior"] == 0) & (h["none"] == 1)).sum())
            p = stats.binomtest(b, b + c).pvalue if b + c else 1.0
            INFO_ROWS[ck, dom] = {"target": tg, "closure": cl, "n_undet": len(h["none"]), "b": b, "c": c, "p": p}
            parts.append(f"{ck} n {len(h['none']):4d} {100 * h['none'].mean():5.1f}/{100 * h['prior'].mean():5.1f}/"
                         f"{100 * h['prior_shuf'].mean():5.1f} {b}:{c} p {p:.2g}")
        print(f"    {dom:13s} {INFO_CLASS[dom]:10s} target {100 * tg:5.1f}%  closure {100 * cl:5.1f}% | " + " | ".join(parts))

print("\n== families (domain-equal means; pooled items = discordant b:c summed over domains and seeds, item-weighted)")
for ck in CKS + ["both"]:
    for fam in FAMILY:
        rows = [res[k] for k in KEYS if res[k]["family"] == fam and ck in (k[0], "both")]
        print(f"  {ck:4s} {fam:20s} {pool(rows)}")
        for task in ("off", "cmp"):
            sub = [r for r in rows if r["task"] == task]
            if sub and len(sub) < len(rows):
                print(f"  {ck:4s}   {fam[:12]}/{task:5s}     {pool(sub)}")
    for ic in ("offset", "cmp-order", "cmp-anchor"):
        print(f"  {ck:4s} {'ALL ' + ic:20s} {pool([res[k] for k in KEYS if INFO_CLASS[k[1]] == ic and ck in (k[0], 'both')])}")

print("\n== competence band of none (floor <= chance + 5, ceiling >= 95, middle otherwise)")
for ck in CKS + ["both"]:
    for band in ("floor", "middle", "ceiling"):
        rows = [res[k] for k in KEYS if res[k]["band"] == band and ck in (k[0], "both")]
        if rows:
            print(f"  {ck:4s} {band:8s} {pool(rows)}")
            if ck != "both":
                print("       " + ", ".join(f"{r['domain']} {r['P-N']['d']:+.1f}{sig(r['P-N']).strip()}" for r in rows))

print("\n== gain vs competence (F25 / F26 inverted U). x = chance-normalised none accuracy, y = chance-normalised P-N (x100)")
X = np.array([res[k]["x"] for k in KEYS]) / 100
Y = np.array([res[k]["y"] for k in KEYS])
TASK = np.array([res[k]["task"] for k in KEYS])
DOMK = np.array([k[1] for k in KEYS])
NONE = np.array([res[k]["mean"]["none"] for k in KEYS])
CHANCE = np.array([res[k]["chance"] for k in KEYS])
ABOVE = NONE > CHANCE + 5   # none above the measurement floor
print("  F25 bins of x (cells by class: off = offset, target carries no answers; cmp = comparison, target implies >= 94% of answers):")
for lab_m, mm0 in (("all", np.ones(len(KEYS), bool)), ("offset", TASK == "off"), ("offset above floor", (TASK == "off") & ABOVE),
                   ("comparison", TASK == "cmp")):
    out = []
    for lo, hi in BINS:
        m = mm0 & (100 * X >= lo) & (100 * X < hi)
        out.append(f"{Y[m].mean():+6.1f} (n {m.sum():2d}, med {np.median(Y[m]):+5.1f})" if m.sum() else f"{'-':>22s}")
    print(f"    {lab_m:19s} " + "  ".join(out))
print(f"    bins: {', '.join(f'<{hi}' if lo < -1e8 else f'>={lo}' if hi > 1e8 else f'{lo}-{hi}' for lo, hi in BINS)}; middle bins"
      " (25-75): " + ", ".join(f"{k[0]}/{k[1]} {res[k]['P-N']['d']:+.1f}{'R' if res[k]['P-N']['robust'] else ''}"
                               for k, xx in zip(KEYS, X) if 0.25 <= xx < 0.75))


def ushape(m, label, loo=False):
    x, y, dk = X[m], Y[m], DOMK[m]
    rho, p = stats.spearmanr(x, y)
    rho_u, p_u = stats.spearmanr(x * (1 - x), y)
    b2, b1, b0 = np.polyfit(x, y, 2)
    doms = sorted(set(dk))
    boot = []
    rng = np.random.default_rng(1)
    for _ in range(2000):
        pick = rng.choice(len(doms), len(doms))
        idx = np.concatenate([np.where(dk == doms[j])[0] for j in pick])
        if len(set(x[idx])) > 3:
            boot.append(np.polyfit(x[idx], y[idx], 2)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    L, R = x < 0.5, x >= 0.5

    def side(k):
        if k.sum() < 5:
            return f"n {k.sum()} (too few)"
        s_ = stats.spearmanr(x[k], y[k])
        return f"rho {s_[0]:+.2f} (p {s_[1]:.3f}, n {k.sum()})"
    vx = -b1 / (2 * b2) if b2 < 0 else np.nan
    print(f"  {label:22s} n={m.sum():2d}  Spearman(x, y) {rho:+.2f} (p {p:.3f})  Spearman(x(1-x), y) {rho_u:+.2f} (p {p_u:.3f})  "
          f"quadratic {b2:+.1f} [{lo:+.1f}, {hi:+.1f}] vertex x {f'{vx:.2f}' if 0 <= vx <= 1 else '-'}  "
          f"x<0.5: {side(L)}  x>=0.5: {side(R)}")
    if loo:   # leave one domain out (both checkpoints of it): range of the U statistics
        st = []
        for dm in doms:
            k = dk != dm
            st.append((stats.spearmanr(x[k] * (1 - x[k]), y[k])[0], np.polyfit(x[k], y[k], 2)[0],
                       stats.spearmanr(x[k & L], y[k & L])[0], stats.spearmanr(x[k & R], y[k & R])[0]))
        st = np.array(st)
        print(f"  {'':22s} leave one domain out: Spearman(x(1-x), y) {st[:, 0].min():+.2f}..{st[:, 0].max():+.2f}  quadratic "
              f"{st[:, 1].min():+.1f}..{st[:, 1].max():+.1f}  x<0.5 rho {st[:, 2].min():+.2f}..{st[:, 2].max():+.2f}  x>=0.5 rho "
              f"{st[:, 3].min():+.2f}..{st[:, 3].max():+.2f}")


for ck in CKS:
    ushape(np.array([k[0] == ck for k in KEYS]), f"{ck} all")
ALLM = np.ones(len(KEYS), bool)
ushape(ALLM, "both all", loo=True)
ushape(TASK == "off", "both offset")
ushape((TASK == "off") & ABOVE, "both offset above floor")
ushape(TASK == "cmp", "both comparison")
R_ = X >= 0.5
print(f"  composition: x >= 0.5 side {int(R_.sum())} cells = {int((R_ & (TASK == 'cmp')).sum())} comparison + "
      f"{int((R_ & (TASK == 'off')).sum())} offset; x < 0.5 side {int((~R_).sum())} cells = {int((~R_ & (TASK == 'off')).sum())} offset "
      f"({int((~R_ & (TASK == 'off') & ~ABOVE).sum())} at the floor) + {int((~R_ & (TASK == 'cmp')).sum())} comparison; offset cells "
      f"above the floor (no target answers, measurable): {int(((TASK == 'off') & ABOVE).sum())}")

G = np.array([res[k]["P-N"]["d"] for k in KEYS])
hm = (NONE <= 99) & (X >= 0.5)
g = G[hm] / (100 - NONE[hm])
s_ = stats.spearmanr(X[hm], g)
print(f"  headroom: on the falling side (x >= 0.5, none <= 99, n {hm.sum()}, {int((hm & (TASK == 'cmp')).sum())} comparison) the gain as a "
      f"share of the headroom (prior - none) / (100 - none) has Spearman {s_[0]:+.2f} (p {s_[1]:.3f}) with x (raw gain "
      f"{stats.spearmanr(X[hm], G[hm])[0]:+.2f}); median share {np.median(g):+.2f}")
print("  same domain, two checkpoints (the F25 / F26 design; offset domains with none above the floor at >= 1 checkpoint):")
for dom in sorted(d for d in DOMS if INFO_CLASS[d] == "offset"):
    rr = [res[ck, dom] for ck in CKS if (ck, dom) in res]
    if any(r["mean"]["none"] > r["chance"] + 5 for r in rr):
        print("    " + f"{dom:13s} " + "  ->  ".join(
            f"{r['ck']} none {r['mean']['none']:5.1f} P-N {r['P-N']['d']:+5.1f}{'R' if r['P-N']['robust'] else ''} "
            f"P-S {r['P-S']['d']:+5.1f}" for r in rr))

print("\n== did the prior act? CKA of the held-out item states with the TRUE target (mean of L4-16; prompts = 256 test questions)"
      "\n   and the logged training geo loss 1 - CKA against the arm's own target (shuffled for prior_shuf), step 0 -> 210")
for ck in CKS:
    rows = [res[k] for k in KEYS if k[0] == ck]
    dP = np.array([[p - b for p, b in zip(r["ctx"]["prior"], r["ctx"]["base"])] for r in rows])
    dS = np.array([[p - b for p, b in zip(r["ctx"]["prior_shuf"], r["ctx"]["base"])] for r in rows])
    dN = np.array([[p - b for p, b in zip(r["ctx"]["none"], r["ctx"]["base"])] for r in rows])
    gP = np.array([[g[1] < g[0] for g in r["geo"]["prior"]] for r in rows])
    gS = np.array([[g[1] < g[0] for g in r["geo"]["prior_shuf"]] for r in rows])
    print(f"  {ck}: prior ctx CKA - base: median {np.median(dP):+.3f}, > 0 in {int((dP > 0).sum())}/{dP.size} runs; prior_shuf - base median "
          f"{np.median(dS):+.3f} (< 0 in {int((dS < 0).sum())}/{dS.size}); none - base median {np.median(dN):+.3f}; geo loss fell: prior "
          f"{int(gP.sum())}/{gP.size}, prior_shuf {int(gS.sum())}/{gS.size}; shuf_cka max {max(max(r['shuf_cka']) for r in rows):.3f}")
    for r, a, b in zip(rows, dP, gP):
        if (a <= 0).any() or not b.all():
            print(f"     prior not toward the target: {r['domain']} ctx base {np.mean(r['ctx']['base']):.3f} -> prior {np.mean(r['ctx']['prior']):.3f}"
                  f" geo {[f'{g[0]:.3f}->{g[1]:.3f}' for g in r['geo']['prior']]}")
    for r, a in zip(rows, dS):
        if (a >= 0).any():
            print(f"     prior_shuf true-target CKA not below base: {r['domain']} seeds {[f'{v:+.3f}' for v in a]} (shuf_cka {[round(v, 3) for v in r['shuf_cka']]})")
    for r, b in zip(rows, gS):
        if not b.all():
            print(f"     prior_shuf geo loss did not fall: {r['domain']} {[f'{g[0]:.3f}->{g[1]:.3f}' for g in r['geo']['prior_shuf']]}")
    print(f"  {ck} per domain: ctx CKA (true target) base / none / prior / shuf | probe base / none / prior | geo loss at step 210 prior / shuf | shuf_cka")
    for r in sorted(rows, key=lambda r: r["domain"]):
        m = {a: np.mean(r["ctx"][a]) for a in FILES}
        pm = {a: np.mean(r["probe"][a]) for a in FILES}
        print(f"     {r['domain']:15s} {m['base']:.2f} / {m['none']:.2f} / {m['prior']:.2f} / {m['prior_shuf']:.2f} | "
              f"{pm['base']:.2f} / {pm['none']:.2f} / {pm['prior']:.2f} | {np.mean([g[1] for g in r['geo']['prior']]):.3f} / "
              f"{np.mean([g[1] for g in r['geo']['prior_shuf']]):.3f} | {np.mean(r['shuf_cka']):.2f}")

print("\n== controls")
low = [f"{r['ck']}/{r['domain']} {r['train']['none']:.1f}" for r in res.values() if r["train"]["none"] < 95]
print(f"  ceiling: none train accuracy >= 95 in {sum(r['train']['none'] >= 95 for r in res.values())}/{len(res)} cells; below: {', '.join(low)}")
lowp = [f"{r['ck']}/{r['domain']} {r['train']['prior_shuf']:.1f}" for r in res.values() if r["train"]["prior_shuf"] < 90]
print(f"  prior_shuf train accuracy < 90: {', '.join(lowp) if lowp else 'none'}")
na = [f"{r['domain']} {100 * np.mean(r['ans_tr']):.1f}%" for r in res.values() if np.mean(r["ans_tr"]) < 0.999 and r["ck"] == CKS[0]]
print(f"  test questions whose answer is a training answer (n_test_answer_trained / n_test, < 100%): {', '.join(na)}")
BS = 16   # not in the JSONs: ft36 default, make_jobs_ft36.py does not override it (AUDIT.md: bs 16)
for r in res.values():
    r["epochs"] = LAUNCH["steps"] * BS / np.mean(r["n_train"])
below = [r for r in res.values() if r["task"] == "off" and r["mean"]["none"] < r["mean"]["base"]]
print(f"  offset cells where fine-tuning (none) is BELOW the untuned base on held-out questions: {len(below)}/"
      f"{sum(r['task'] == 'off' for r in res.values())}, of them below chance {sum(r['mean']['none'] < r['chance'] for r in below)}: "
      + ", ".join(f"{r['ck']}/{r['domain']} {r['mean']['base']:.1f}->{r['mean']['none']:.1f} ({r['epochs']:.0f} ep)" for r in below))
eo = {d: res[CKS[0], d]["epochs"] for d in DOMS if (CKS[0], d) in res}
print(f"  epochs = {LAUNCH['steps']} steps x bs {BS} / n_train: offset {min(v for d, v in eo.items() if d in OFFSET_DOMAINS):.1f}.."
      f"{max(v for d, v in eo.items() if d in OFFSET_DOMAINS):.0f}, comparison {min(v for d, v in eo.items() if d not in OFFSET_DOMAINS):.1f}.."
      f"{max(v for d, v in eo.items() if d not in OFFSET_DOMAINS):.0f}; offset domains with < 10 epochs: "
      + ", ".join(f"{d} {v:.1f}" for d, v in sorted(eo.items(), key=lambda t: t[1]) if d in OFFSET_DOMAINS and v < 10))
if ("42B", "numbers") in res:
    r = res["42B", "numbers"]
    t, u = r["P-N"], r["P-S"]
    ok = t["robust"] and t["b"] > t["c"] and u["wins"] == 3
    print(f"  positive control numbers @ 42B: none {r['mean']['none']:.1f} prior {r['mean']['prior']:.1f} shuf {r['mean']['prior_shuf']:.1f}; "
          f"P-N {t['d']:+.1f} ({t['wins']}/3, Holm p {t['holm']:.2g}), P-S {u['d']:+.1f} ({u['wins']}/3) -> {'PASS' if ok else 'FAIL'} "
          f"(F10-F14: helix > none at 42B)")

print("\n== domain lists for the next design (ft36b / pretraining manifolds). Per checkpoint: none (mean), P-N (R = robust);"
      "\n   measurable = chance + 5 < none <= 90 (headroom >= 10); near-ceiling = 90 < none < 95; info: offset = target carries"
      "\n   no answers (item order implied by training), cmp-order / cmp-anchor = target implies the held-out answers"
      " (training closure implies some / almost none)")


def cell_txt(ck, dom):
    r = res[ck, dom]
    return f"{ck} {r['mean']['none']:5.1f} {r['P-N']['d']:+5.1f}{'R' if r['P-N']['robust'] else ' '}"


def state(r):
    n = r["mean"]["none"]
    return "floor" if n <= r["chance"] + 5 else "ceiling" if n >= 95 else "near-ceiling" if n > 90 else "measurable"


LISTS = collections.defaultdict(list)
for dom in DOMS:
    st = {ck: state(res[ck, dom]) for ck in CKS if (ck, dom) in res}
    nt = np.mean(res[CKS[0], dom]["n_test"])
    if "measurable" in st.values():
        LISTS["a" if INFO_CLASS[dom] != "offset" else "ab"].append(dom)
    elif set(st.values()) <= {"floor"}:
        LISTS["c-floor"].append(dom)
    elif set(st.values()) <= {"ceiling", "near-ceiling"}:
        LISTS["c-ceiling"].append(dom)
    else:
        LISTS["c-jump"].append(dom)
    res[CKS[0], dom]["n_test_mean"] = nt
for lab, title in (("ab", "(a)+(b) measurable at >= 1 checkpoint AND target carries no answers: geometry, not facts"),
                   ("a", "(a) measurable at >= 1 checkpoint, but the target implies the held-out answers: facts, not geometry"),
                   ("c-floor", "(c) floor at every checkpoint"),
                   ("c-ceiling", "(c) ceiling or near-ceiling at every checkpoint"),
                   ("c-jump", "(c) floor at one checkpoint, (near-)ceiling at the other: the window lies between")):
    print(f"  {title}: {len(LISTS[lab])}")
    for dom in LISTS[lab]:
        r0 = res[CKS[0], dom]
        base_txt = " / ".join(f"{res[ck, dom]['mean']['base']:.1f}" for ck in CKS if (ck, dom) in res)
        print(f"     {dom:14s} {INFO_CLASS[dom]:10s} n_test {r0['n_test_mean']:6.0f} epochs {r0['epochs']:5.1f} | "
              + " | ".join(f"{cell_txt(ck, dom)} {state(res[ck, dom])}" for ck in CKS if (ck, dom) in res)
              + f" | base {base_txt} chance {r0['chance']:.1f}")
for k in KEYS:
    res[k]["info_class"] = INFO_CLASS[k[1]]
    res[k]["info"] = INFO_ROWS.get(k)

if args.json:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    Path(args.json).write_text(json.dumps(clean(list(res.values())), indent=1))
    print(f"\ncells written to {args.json}")
