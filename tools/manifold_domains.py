"""Concept domains for the manifold atlas (tools/manifold_atlas.py): items, prompts and candidate target geometries.

Every domain gives items (strings), a ground-truth coordinate per item and two prompt templates whose LAST token is
the item's last token ("neutral" = "Q: {item}", as in the number experiments; "cued" names the domain first). The
targets are feature matrices (items x features); the atlas measures how well each one is linearly decodable from
the states and against a permutation null. Candidate targets per kind (§5 of report/manifold_survey.md: for a
domain with symmetry group G, the natural target is G's irreps):
  cyclic  Z_n          circle (k = 1), fourier (all k), line (wrong topology, control)
  helix   numbers      helix (periods 2/5/10/100 + linear), line
  linear  ordered      line (index), curve (orthogonal polynomials up to degree 3)
  sphere  sky, cities  sphere (unit vector), harmonics (real spherical harmonics, degree <= 2), plane (lat, long)
  plane   emotions     plane (valence, arousal from Warriner et al. 2013), circumplex (angle on the VA plane)
Data sources: matplotlib CSS4 colours (hue from RGB), Hipparcos via skyfield (stars with IAU proper names),
geonamescache (cities), Warriner et al. 2013 norms (emotions); the rest are fixed sequences written out here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

DATA = Path("/scratch/yh6210/llmadd_torch/data/atlas")

DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
         "phi chi psi omega").split()
NUMWORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen twenty").split() + "thirty forty fifty sixty seventy eighty ninety".split()
NUMWORD_VALUES = list(range(1, 21)) + [30, 40, 50, 60, 70, 80, 90]
ORDINALS = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
            "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()
PLANETS = "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune".split()
ELEMENTS = ("hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon sodium magnesium aluminum "
            "silicon phosphorus sulfur chlorine argon potassium calcium scandium titanium vanadium chromium manganese "
            "iron cobalt nickel copper zinc gallium germanium arsenic selenium bromine krypton").split()
# US presidents in office order ("-" = left out: the surnames shared by two presidents, Adams, Harrison, Johnson,
# Cleveland, Roosevelt and Bush, are dropped so that every name has one position)
PRESIDENTS = ("Washington - Jefferson Madison Monroe - Jackson Van_Buren - Tyler Polk Taylor Fillmore Pierce "
              "Buchanan Lincoln - Grant Hayes Garfield Arthur - - - McKinley - Taft Wilson Harding "
              "Coolidge Hoover - Truman Eisenhower Kennedy - Nixon Ford Carter Reagan - Clinton - Obama Trump Biden").split()
# pitch classes (chromatic index) and the 12 major keys as conventionally spelled
PITCH = [("C", 0), ("C#", 1), ("D", 2), ("D#", 3), ("E", 4), ("F", 5), ("F#", 6), ("G", 7), ("G#", 8), ("A", 9),
         ("A#", 10), ("B", 11)]
KEYS = [("C", 0), ("Db", 1), ("D", 2), ("Eb", 3), ("E", 4), ("F", 5), ("F#", 6), ("G", 7), ("Ab", 8), ("A", 9),
        ("Bb", 10), ("B", 11)]
COLOURS = ("red orange yellow gold chartreuse lime green olive teal turquoise cyan aqua blue navy indigo violet purple "
           "magenta orchid plum pink crimson coral salmon tomato maroon").split()
EMOTIONS = ("happy sad angry afraid calm excited bored relaxed anxious content depressed elated furious gloomy joyful "
            "lonely nervous peaceful proud scared serene tense tired upset cheerful delighted enthusiastic frustrated "
            "grateful guilty hopeful hostile irritated jealous miserable optimistic pleased sleepy surprised terrified "
            "thrilled worried ashamed disgusted ecstatic embarrassed energetic inspired lively panicked satisfied "
            "tranquil triumphant uneasy weary alert amused annoyed").split()


# ------------------------------------------------------------------------------------------------ target features

def _std(F):
    F = np.asarray(F, dtype=np.float64)
    F = F - F.mean(0)
    s = F.std(0)
    return F / np.where(s > 0, s, 1)


def fourier(idx, n, ks):
    th = 2 * np.pi * np.asarray(idx)[:, None] / n
    return _std(np.concatenate([np.cos(th * k) for k in ks] + [np.sin(th * k) for k in ks if 2 * k != n], 1))


def line(v):
    return _std(np.asarray(v, dtype=np.float64)[:, None])


def curve(v, deg=3):
    x = np.asarray(v, dtype=np.float64)
    x = 2 * (x - x.min()) / (np.ptp(x) or 1) - 1
    return _std(np.polynomial.legendre.legvander(x, deg)[:, 1:])


def helix(v, periods=(2, 5, 10, 100)):
    v = np.asarray(v, dtype=np.float64)
    cols = [v]
    for T in periods:
        cols += [np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)]
    return _std(np.stack(cols, 1))


def unit_vec(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def harmonics(lat_deg, lon_deg):
    """Real spherical harmonics of degree 1 and 2 (8 functions) on the unit sphere."""
    x, y, z = unit_vec(lat_deg, lon_deg).T
    return _std(np.stack([x, y, z, x * y, y * z, x * z, x * x - y * y, 3 * z * z - 1], 1))


# ------------------------------------------------------------------------------------------------ data-backed lists

def _colours():
    import colorsys
    from matplotlib.colors import CSS4_COLORS, to_rgb
    out = []
    for c in COLOURS:
        h, s, v = colorsys.rgb_to_hsv(*to_rgb(CSS4_COLORS[c]))
        if s > 0.3 and v > 0.25:
            out.append((c, 360 * h))
    return out


def _stars(max_mag=2.6):
    seen, out = set(), []
    with open(DATA / "stars.csv") as f:
        for r in csv.DictReader(f):
            if r["hip"] in seen or float(r["magnitude"]) > max_mag or " " in r["name"]:
                continue
            seen.add(r["hip"])
            out.append((r["name"], float(r["dec_degrees"]), float(r["ra_degrees"])))
    return out


def _cities(n=60, per_country=2):
    import geonamescache
    cs = sorted(geonamescache.GeonamesCache().get_cities().values(), key=lambda c: -c["population"])
    out, cnt, names = [], {}, set()
    for c in cs:
        nm = c["name"]
        if not nm.isascii() or nm in names or cnt.get(c["countrycode"], 0) >= per_country:
            continue
        cnt[c["countrycode"]] = cnt.get(c["countrycode"], 0) + 1
        names.add(nm)
        out.append((nm, float(c["latitude"]), float(c["longitude"])))
        if len(out) == n:
            break
    return out


def _emotions():
    va = {}
    with open(DATA / "warriner.csv") as f:
        for r in csv.DictReader(f):
            va[r["Word"]] = (float(r["V.Mean.Sum"]), float(r["A.Mean.Sum"]))
    return [(w, *va[w]) for w in EMOTIONS if w in va]


# ------------------------------------------------------------------------------------------------ domains

def domains() -> dict:
    D = {}

    def add(name, kind, items, cued, targets, coord):
        D[name] = dict(kind=kind, items=list(items), cued=cued, targets=targets, coord=np.asarray(coord))

    x = np.arange(10, 100)
    add("numbers", "helix", [str(v) for v in x], "The number {}", {"helix": helix(x), "line": line(x)}, x)
    v = np.array(NUMWORD_VALUES)
    add("number_words", "helix", NUMWORDS, "The number {}", {"helix": helix(v, (2, 5, 10)), "line": line(v)}, v)
    i = np.arange(1, 21)
    add("ordinals", "helix", ORDINALS, "It came {}", {"helix": helix(i, (2, 5, 10)), "line": line(i)}, i)
    y = np.arange(1900, 2000)
    add("years", "helix", [str(t) for t in y], "In the year {}", {"helix": helix(y - 1900, (10, 100)), "line": line(y)}, y)
    for name, items, cued in (("days", DAYS, "The day was {}"), ("months", MONTHS, "The month of {}")):
        n = len(items); i = np.arange(n)
        add(name, "cyclic", items, cued, {"circle": fourier(i, n, (1,)), "fourier": fourier(i, n, range(1, n // 2 + 1)),
                                          "line": line(i)}, i)
    h = np.arange(24)
    add("hours", "cyclic", [f"{t}:00" for t in h], "The time was {}",
        {"circle": fourier(h, 24, (1,)), "fourier": fourier(h, 24, range(1, 13)), "base10": helix(h, (2, 5, 10)),
         "line": line(h)}, h)
    i = np.arange(26)
    add("letters", "linear", [chr(65 + k) for k in i], "The letter {}",
        {"line": line(i), "curve": curve(i), "circle": fourier(i, 26, (1,))}, i)
    i = np.arange(24)
    add("greek", "linear", GREEK, "The Greek letter {}", {"line": line(i), "curve": curve(i), "circle": fourier(i, 24, (1,))}, i)
    for name, items, cued in (("planets", PLANETS, "The planet {}"), ("elements", ELEMENTS, "The element {}")):
        i = np.arange(len(items))
        add(name, "linear", items, cued, {"line": line(i), "curve": curve(i)}, i)
    pres = [(p.replace("_", " "), k) for k, p in enumerate(PRESIDENTS, 1) if p != "-"]
    i = np.array([k for _, k in pres])
    add("presidents", "linear", [p for p, _ in pres], "President {}", {"line": line(i), "curve": curve(i)}, i)
    for name, items, cued in (("pitch", PITCH, "The note {}"), ("keys", KEYS, "A piece in the key of {}")):
        c = np.array([k for _, k in items])
        fifths = (7 * c) % 12   # position on the circle of fifths
        add(name, "cyclic", [n for n, _ in items], cued,
            {"chromatic": fourier(c, 12, (1,)), "fifths": fourier(fifths, 12, (1,)), "fourier": fourier(c, 12, range(1, 7)),
             "line": line(c)}, c)
    col = _colours(); hue = np.array([hh for _, hh in col])
    add("colours", "cyclic", [c for c, _ in col], "The colour {}",
        {"hue": _std(np.stack([np.cos(np.radians(hue)), np.sin(np.radians(hue))], 1)), "line": line(hue)}, hue)
    st = _stars(); dec = np.array([s[1] for s in st]); ra = np.array([s[2] for s in st])
    add("stars", "sphere", [s[0] for s in st], "In the night sky, the star {}",
        {"sphere": _std(unit_vec(dec, ra)), "harmonics": harmonics(dec, ra), "plane": _std(np.stack([dec, ra], 1))},
        np.stack([dec, ra], 1))
    ci = _cities(); lat = np.array([c[1] for c in ci]); lon = np.array([c[2] for c in ci])
    add("cities", "sphere", [c[0] for c in ci], "The city of {}",
        {"sphere": _std(unit_vec(lat, lon)), "harmonics": harmonics(lat, lon), "plane": _std(np.stack([lat, lon], 1))},
        np.stack([lat, lon], 1))
    em = _emotions(); V = np.array([e[1] for e in em]); A = np.array([e[2] for e in em])
    ang = np.arctan2(_std(A[:, None])[:, 0], _std(V[:, None])[:, 0])
    add("emotions", "plane", [e[0] for e in em], "I feel {}",
        {"plane": _std(np.stack([V, A], 1)), "valence": line(V), "arousal": line(A),
         "circumplex": _std(np.stack([np.cos(ang), np.sin(ang)], 1))}, np.stack([V, A], 1))
    return D


if __name__ == "__main__":
    for k, d in domains().items():
        print(f"{k:13s} {d['kind']:7s} n={len(d['items']):3d}  targets {list(d['targets'])}  e.g. {d['items'][:4]}")


# ------------------------------------------------------------------------------------------------ test sets
# One task family per domain, scored by choosing the answer among candidates (summed log-prob of " candidate"
# after a 3-shot prompt). offset = "k steps after x" over the domain's own items (candidates = all items, chance 1/n);
# pair = a two-way comparison that needs the domain's geometry (candidates = the two options, chance 1/2).
OFFSET = {
    "numbers": ("{x} + {k} =", "add", range(10, 50)),
    "number_words": ("{x} plus {k} is", "linear", range(1, 8)),
    "ordinals": ("{k} places after {x} is", "linear", range(1, 6)),
    "years": ("{k} years after {x} is", "linear", range(1, 21)),
    "days": ("{k} days after {x} is", "cyclic", range(1, 7)),
    "months": ("{k} months after {x} is", "cyclic", range(1, 12)),
    "hours": ("{k} hours after {x} is", "cyclic", range(1, 13)),
    "letters": ("{k} letters after {x} is", "linear", range(1, 6)),
    "greek": ("in the Greek alphabet, {k} letters after {x} is", "linear", range(1, 5)),
    "planets": ("counting outward from the Sun, {k} planets after {x} is", "linear", range(1, 4)),
    "elements": ("in the periodic table, {k} elements after {x} is", "linear", range(1, 4)),
    "pitch": ("{k} semitones above {x} is", "cyclic", range(1, 12)),
    "keys": ("{k} semitones above the key of {x} is the key of", "cyclic", range(1, 12)),
}
PAIR = {
    "presidents": "Who was US president earlier, {b} or {c}? Answer:",
    "colours": "Which colour is closer in hue to {a}, {b} or {c}? Answer:",
    "stars": "Which star is closer to {a} in the night sky, {b} or {c}? Answer:",
    "cities": "Which city is closer to {a}, {b} or {c}? Answer:",
    "emotions": "Which feeling is more pleasant, {b} or {c}? Answer:",
}
WORD_NUM = dict(zip(("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                     "seventeen eighteen nineteen twenty").split(), range(1, 21)))


def _sphere_dist(p, q):
    a, b = unit_vec(*np.asarray(p)[:, None]), unit_vec(*np.asarray(q)[:, None])
    return float(np.degrees(np.arccos(np.clip((a * b).sum(), -1, 1))))


def tasks(D: dict, n: int = 60, shots: int = 3, seed: int = 0) -> dict:
    """{domain: [(prompt, candidates, answer_index), ...]}; the shots come from the same generator."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, d in D.items():
        items, nI = d["items"], len(d["items"])
        gen = []
        if name in OFFSET:
            tmpl, mode, ks = OFFSET[name]
            ks = list(ks)
            while len(gen) < n + shots:
                i, k = int(rng.integers(nI)), int(rng.choice(ks))
                if mode == "add":
                    a = int(d["coord"][i])
                    if a + k > 99:
                        continue
                    gen.append((tmpl.format(x=a, k=k), items.index(str(a + k))))
                    continue
                j = (i + k) % nI if mode == "cyclic" else i + k
                if j >= nI:
                    continue
                if name == "number_words" and items[j] not in WORD_NUM:   # stay within one..twenty
                    continue
                kw = str(k) if name in ("years", "hours") else list(WORD_NUM)[k - 1]
                q = tmpl.format(x=items[i], k=kw)
                if k == 1:   # "one letters after" -> "one letter after"
                    w = q.split(" ")
                    at = w.index(kw) + 1
                    if at < len(w) and w[at].endswith("s") and w[at] != "is":
                        w[at] = w[at][:-1]
                    q = " ".join(w)
                gen.append((q, j))
            cands = items
            make = lambda g: (g[0], cands, g[1])
        else:
            tmpl = PAIR[name]
            c = d["coord"]
            while len(gen) < n + shots:
                a, b, e = (int(t) for t in rng.choice(nI, 3, replace=False))
                if name == "presidents":
                    if abs(c[b] - c[e]) < 4:
                        continue
                    ans = 0 if c[b] < c[e] else 1
                elif name == "emotions":
                    if abs(c[b][0] - c[e][0]) < 2:
                        continue
                    ans = 0 if c[b][0] > c[e][0] else 1
                else:
                    if name == "colours":
                        dist = lambda u, v: min(abs(c[u] - c[v]), 360 - abs(c[u] - c[v]))
                    else:
                        dist = lambda u, v: _sphere_dist(c[u], c[v])
                    db, de = dist(a, b), dist(a, e)
                    if abs(db - de) < (30 if name == "colours" else 25):
                        continue
                    ans = 0 if db < de else 1
                gen.append((tmpl.format(a=items[a], b=items[b], c=items[e]), (items[b], items[e]), ans))
            make = lambda g: (g[0], list(g[1]), g[2])
        rows = [make(g) for g in gen]
        shot_txt = "".join(f"{p} {cs[ai]}\n" for p, cs, ai in rows[:shots])
        out[name] = [(shot_txt + p, cs, ai) for p, cs, ai in rows[shots:]]
    return out
