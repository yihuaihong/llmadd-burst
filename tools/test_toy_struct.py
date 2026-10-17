"""Tests for tools/toy_struct.py (run before any job; outputs under --out).

1. known answer: --task mod reproduces tools/toy_grok.py (same init, split, per-step losses / accuracies / CKA for
   the first 200 steps, arms none, w0, w100, always_shuf), max |diff| <= 1e-6 (bitwise expected: 1 CPU thread, same device).
2. group axioms, targets, CKA matrix, split disjointness, closure (incl. known-answer cases and asserts that must fire).
2b. the decision rules on synthetic cells with known verdicts (partial rescue, rescue, floor, overlap).
3. smoke: every arm of s4 and dim at both fractions, 300 steps on CPU; the prior is read (CKA to the own target rises
   above none's), the window switches off, the error-share metrics are in range.
4. timing: ms/step on this CPU (1 thread) per task / fraction.

    python tools/test_toy_struct.py --out <scratch dir> [--skip_known]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_struct as ts  # noqa: E402

PY = sys.executable


def known_answer(out: Path) -> None:
    arms = "none,w0,w100,always_shuf"
    common = ["--steps", "200", "--eval_every", "1", "--seeds", "0", "--arms", arms]
    # one intra-op thread for both: with 2+ CPU threads toy_grok.py itself is not bitwise reproducible run to run
    # (measured 2026-10-16: two identical toy_grok runs differ from step 1, max |diff| 7.8e-4 by step 60); with 1
    # thread both scripts are deterministic and must agree bitwise.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    procs, t = [], time.time()
    for name, cmd in (("grok", [PY, str(HERE / "toy_grok.py"), "--out", str(out / "ka_grok")] + common),
                      ("struct", [PY, str(HERE / "toy_struct.py"), "--task", "mod", "--device", "cpu", "--out", str(out / "ka_struct")] + common)):
        if not any((out / f"ka_{name}").glob("run_*.json")):
            procs.append((name, subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
    for name, pr in procs:
        err = pr.communicate()[1]
        assert pr.returncode == 0, err[-2000:]
        print(f"  {name}: {time.time() - t:.0f}s")
    worst, n_exact, n_all = 0.0, 0, 0
    for arm in arms.split(","):
        g = json.loads((out / "ka_grok" / f"run_{arm}_seed0.json").read_text())
        s = json.loads((out / "ka_struct" / f"run_{arm}_seed0.json").read_text())
        assert len(g["curve"]) == len(s["curve"]) == 200, (len(g["curve"]), len(s["curve"]))
        for cg, cs in zip(g["curve"], s["curve"]):
            for k in ("step", "ce", "train", "test", "cka_circle", "fourier_top5"):
                d = abs(cg[k] - cs[k])
                worst = max(worst, d); n_all += 1; n_exact += d == 0
        for k in ("t_mem", "t_grok", "final_test", "seed", "frac", "lam", "width", "p", "freqs"):
            assert g[k] == s[k], (arm, k, g[k], s[k])
        print(f"  {arm}: ce step0 {g['curve'][0]['ce']:.6f} / {s['curve'][0]['ce']:.6f}, step199 {g['curve'][-1]['ce']:.6f} / "
              f"{s['curve'][-1]['ce']:.6f}; test@199 {g['curve'][-1]['test']:.4f} / {s['curve'][-1]['test']:.4f}")
    # the arms must differ from each other (the geometry loss acts), else the identity is vacuous
    ce = {a: json.loads((out / "ka_struct" / f"run_{a}_seed0.json").read_text())["curve"][150]["ce"] for a in arms.split(",")}
    assert len(set(ce.values())) == len(ce), ce
    s_w100 = json.loads((out / "ka_struct" / "run_w100_seed0.json").read_text())["curve"]
    s_none = json.loads((out / "ka_struct" / "run_none_seed0.json").read_text())["curve"]
    assert all(a["ce"] == b["ce"] for a, b in zip(s_w100[:100], s_none[:100])) and s_w100[101]["ce"] != s_none[101]["ce"], "w100 window"
    print(f"KNOWN ANSWER mod: {n_exact}/{n_all} values bitwise equal, max |diff| {worst:.2e}")
    assert worst <= 1e-6, worst


def structure() -> None:
    D = ts.build_task("s4")
    mats, mul, sgn, coset = ts.cube_group()
    # spot-check the convention on one pair: a = 90 deg about z, b = 90 deg about x; "a then b" = R_b R_a
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]); Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    key = {tuple(M.ravel()): i for i, M in enumerate(mats)}
    a, b = key[tuple(Rz.ravel())], key[tuple(Rx.ravel())]
    assert (mats[mul[a, b]] == Rx @ Rz).all() and not (mats[mul[a, b]] == Rz @ Rx).all()
    v = np.array([1, 0, 0])
    assert (mats[mul[a, b]] @ v == Rx @ (Rz @ v)).all()          # x -> y (by a) -> z (by b)
    assert (mats[mul[a, b]] @ v == np.array([0, 0, 1])).all()
    print(f"  s4: 24 elements, {int((mul != mul.T).sum())} / 576 non-commuting cells, {int((sgn == -1).sum())} odd, "
          f"V4 cosets {np.bincount(coset).tolist()}")
    for task in ("s4", "dim"):
        D = ts.build_task(task)
        for seed in range(5):
            T, info = ts.seed_targets(D, seed)
            ts.check_targets(D, T, info)
            names, M = ts.target_table(D, T)
            assert np.allclose(M, M.T) and np.allclose(np.diag(M), 1)
            assert 0 <= M.min() and M.max() <= 1 + 1e-9
            for frac in (0.3, 0.6):
                perm, tr, te = ts.split(D, frac, seed)
                assert (perm[: len(tr)] == tr).all() and len(set(tr) | set(te)) == len(D["X"])
                if task == "s4":   # question split = toy_grok's draw
                    assert len(tr) == int(frac * len(D["X"])) and (perm == np.random.default_rng(seed).permutation(576)).all()
                else:              # class split: whole relation classes
                    assert ts.split_mode("dim") == "class" and abs(len(tr) - frac * len(D["X"])) <= 4
                    assert not (set(D["cls"][tr]) & set(D["cls"][te]))
                rows_tr = {tuple(r) for r in D["X"][tr]}
                assert all(tuple(r) not in rows_tr for r in D["X"][te])
                c = ts.closure(D, tr, te)
                assert c["closure"] == 1.0, (task, seed, frac, c)
                fm = ts.fresh_mask(D, tr, te)
                trs = {tuple(r) for r in D["X"][tr, :3]}
                mates = [bool(_phrasings(D, task, i) - {tuple(D["X"][i, :3])} & trs) for i in te]   # second path
                assert (fm == ~np.array(mates)).all(), (task, seed, frac)
                if task == "dim":
                    assert fm.all()
        print(f"  {task}: targets / ranks / CKA matrix / splits / closure / fresh-mask asserts pass for seeds 0-4 x frac 0.3, 0.6")
    # relation classes against an independent enumeration of each relation's phrasings
    for task in ("s4", "dim"):
        D = ts.build_task(task)
        groups = {}
        for i in range(len(D["X"])):
            groups.setdefault(frozenset(_phrasings(D, task, i)), set()).add(int(D["cls"][i]))
        assert all(len(v) == 1 for v in groups.values()) and len(groups) == int(D["cls"].max()) + 1
        sizes = np.bincount(np.bincount(D["cls"])).tolist()
        print(f"  {task}: {len(groups)} relation classes, size histogram {sizes}")
        assert sizes == ([0, 0, 1, 0, 171] if task == "dim" else [0, 480, 48])
    # the question split stays available for dim and is toy_grok's draw
    Dd = ts.build_task("dim")
    p, tr, te = ts.split(Dd, 0.6, 0, "question")
    assert (p == np.random.default_rng(0).permutation(686)).all() and len(tr) == 411
    # known-answer closure cases
    D = ts.build_task("s4")
    allc = [(a, b) for a in range(24) for b in range(24)]
    e = int(np.where((mats.reshape(24, 9) == np.eye(3).ravel()).all(1))[0][0])
    k = ts.s4_closure(D["mul"], [c for c in allc if c != (3, 5)])
    assert (3, 5) in k
    k = ts.s4_closure(D["mul"], [(e, b) for b in range(24)])
    assert k == {(e, b) for b in range(24)}, len(k)                  # the identity row alone implies nothing else
    Dd = ts.build_task("dim")
    E = Dd["E"]; ki = {tuple(x): i for i, x in enumerate(E)}
    ia, ib = ki[(1, 0, 0)], ki[(0, 1, 0)]
    i0 = [i for i in range(len(Dd["X"])) if tuple(Dd["X"][i, :3]) == (ia, 27, ib)]
    rest = np.array([i for i in range(len(Dd["X"])) if i not in i0])
    # one training relation e_a + e_b = e_c: exactly its triangle (b*a, c/a, c/b) is implied
    der = []
    A = np.zeros((1, 27)); A[0, ia] += 1; A[0, ib] += 1; A[0, Dd["Y"][i0[0]]] -= 1
    for i in rest:
        a_, op, b_, _ = Dd["X"][i]
        r = np.zeros(27); r[a_] += 1; r[b_] += 1 if op == 27 else -1; r[Dd["Y"][i]] -= 1
        if np.linalg.matrix_rank(np.vstack([A, r])) == 1:
            der.append((a_, op, b_))
    assert sorted(der) == sorted([(ib, 27, ia), (ki[(1, 1, 0)], 28, ia), (ki[(1, 1, 0)], 28, ib)]), der
    # asserts that must fire
    D2 = copy.deepcopy(D); D2["X"] = np.vstack([D2["X"], D2["X"][:1]]); D2["Y"] = np.concatenate([D2["Y"], D2["Y"][:1]])
    for bad, msg in ((lambda: ts.split(D2, 0.3, 0), "duplicate"),
                     (lambda: ts.s4_closure(_corrupt(D["mul"]), allc[:400]), "derived")):
        try:
            bad()
        except AssertionError as ex:
            assert msg in str(ex), ex
        else:
            raise SystemExit(f"assert did not fire: {msg}")
    print("  closure known answers (one missing cell derived; identity row implies nothing; one dim relation implies "
          "exactly its triangle) and negative asserts pass")
    # the exact Gehan / Mann-Whitney test: known answers, and scipy's exact MWU as a second path (untied data)
    from scipy.stats import mannwhitneyu
    inf = float("inf")
    assert abs(ts.mwu_p([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) - 1 / 252) < 1e-12
    assert ts.mwu_p([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]) == 1.0 and ts.mwu_p([inf] * 5, [inf] * 5) == 1.0
    assert abs(ts.mwu_p([100, 200, 300, 400, 500], [inf] * 5) - 1 / 252) < 1e-12
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rng.permutation(1000)[:10].reshape(2, 5).tolist()
        assert abs(ts.mwu_p(x, y) - mannwhitneyu(x, y, alternative="less", method="exact").pvalue) < 1e-9
    p45 = ts.mwu_p([100, 200, 300, 400, inf], [inf] * 5)
    print(f"  exact MWU / Gehan p: separated 5 vs 5 = {1 / 252:.4f}; 4/5 grok vs 0/5 = {p45:.4f}; matches scipy exact on 20 random untied cases")


def _phrasings(D: dict, task: str, i: int) -> set:
    """All questions (a, op, b) / (a, b) stating the same relation as question i (independent of D['cls'])."""
    x = D["X"][i]
    if task == "s4":
        a, b = int(x[0]), int(x[1])
        return {(a, b, 24), (b, a, 24)} if D["mul"][a, b] == D["mul"][b, a] else {(a, b, 24)}
    a, op, b = int(x[0]), int(x[1]), int(x[2]); c = int(D["Y"][i])
    u, v, z = (a, b, c) if op == 27 else (c, b, a)   # e_u + e_v = e_z
    return {(u, 27, v), (v, 27, u), (z, 28, v), (z, 28, u)}


def _fake_run(task, arm, seed, curve_test, steps=1000, every=100, frac=0.6, target=None):
    """A synthetic run JSON with the given held-out accuracy per eval (for the summary's known answers)."""
    tg = ["R", "R_sign", "absR", "specrand", "shuf"] if task == "s4" else ["expo", "quot", "onehot", "specrand", "specrand_oh", "shuf"]
    sts = list(range(0, steps, every)) + ([steps - 1] if (steps - 1) % every else [])
    assert len(sts) == len(curve_test)
    key = "coset_share" if task == "s4" else "drop_share"
    curve = [{"step": s, "test": float(t), "train": 1.0, "n_wrong": 100, key: 0.1, **{f"cka_{k}": 0.5 for k in tg}}
             for s, t in zip(sts, curve_test)]
    return {"task": task, "arm": arm, "target": target if target is not None else (None if arm == "none" else arm.split("@")[0]),
            "window": None, "seed": seed, "lam": 1.0, "frac": frac, "steps": steps, "eval_every": every, "split": "question",
            "t_mem": 0, "t_grok": next((s for s, t in zip(sts, curve_test) if t >= 0.95), None),
            "final_test": float(curve_test[-1]), "final_train": 1.0, "stopped_at": None, "chance_majority": 0.03, "curve": curve}


def summary_known_answers(out: Path) -> None:
    """The decision rules on synthetic cells with known verdicts (incl. the audit's partial-rescue pattern)."""
    import shutil
    rng = np.random.default_rng(0)
    n_ev = 11   # steps 1000, eval every 100: 0..900 + 999
    flat = lambda lvl: np.clip(lvl + 0.01 * rng.standard_normal(n_ev), 0, 1)
    grok_at = lambda k: np.array([0.02] * k + [1.0] * (n_ev - k))
    cells = {
        # partial rescue: R / R_sign plateau ~0.7 (never 0.95), specrand / none ~0.01: t_grok all censored -> PASS via AUC
        "partial": {"none": lambda s: flat(0.01), "R": lambda s: flat(0.7), "R_sign": lambda s: flat(0.68),
                    "specrand": lambda s: flat(0.01)},
        # rescue: R / R_sign grok in 4/5 seeds, specrand / none never
        "rescue": {"none": lambda s: flat(0.01), "R": lambda s: grok_at(3 + s) if s < 4 else flat(0.02),
                   "R_sign": lambda s: grok_at(4 + s) if s < 4 else flat(0.02), "specrand": lambda s: flat(0.01)},
        # floor: everything at chance, R a hair above specrand on every seed (p 0.004 but diff < AUC_MIN)
        "floor": {"none": lambda s: flat(0.01), "R": lambda s: flat(0.04), "R_sign": lambda s: flat(0.04),
                  "specrand": lambda s: flat(0.01)},
        # overlap: both generalise partly, no separation
        "overlap": {"none": lambda s: flat(0.01), "R": lambda s: flat(0.4 + 0.1 * (s % 3)), "R_sign": lambda s: flat(0.45),
                    "specrand": lambda s: flat(0.42 + 0.1 * ((s + 1) % 3))},
    }
    want = {"partial": ("PASS (AUC)", "PASS: group", "PARTIAL RESCUE"), "rescue": ("PASS (t_grok + AUC)", "PASS: group", "RESCUE"),
            "floor": ("floor: uninformative", "uninformative (floor", "floor: uninformative"),
            "overlap": ("not shown at n=5", "not shown at n=5", "PARTIAL RESCUE")}
    for name, arms in cells.items():
        d = out / f"summary_ka_{name}"
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        for arm, f in arms.items():
            for s in range(5):
                (d / f"run_{arm}_seed{s}.json").write_text(json.dumps(_fake_run("s4", arm, s, f(s))))
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            ts.summarize(d)
        S = json.loads((d / "summary.json").read_text())
        t = S["tests"]["R<specrand"]
        # the budget is 1000 < LATE here, so (a) cannot qualify: re-read (a) as if it did, to test its verdict logic
        a_q = ts.rule_a(S["tests"], "s4", True)
        print(f"  {name}: R<specrand {t['verdict']} (t_grok p {t['grok']['p']:.3f}, AUC {t['auc']['med_a']:.3f} vs "
              f"{t['auc']['med_b']:.3f} p {t['auc']['p']:.3f}); rule (a) if qualifying: {a_q}; rule (c): {S['rule_c']}")
        assert t["verdict"] == want[name][0], (name, t["verdict"])
        assert a_q.startswith(want[name][1]) or want[name][1] in a_q, (name, a_q)
        assert want[name][2] in S["rule_c"], (name, S["rule_c"])
        assert S["rule_a"].startswith("untestable"), S["rule_a"]   # budget < LATE: never a negative
    # the audit's scenario in numbers: 4/5 grok vs 0/5 passes at ALPHA 0.025; 3/5 does not
    inf = float("inf")
    assert ts.mwu_p([1, 2, 3, 4, inf], [inf] * 5) < ts.ALPHA < ts.mwu_p([1, 2, 3, inf, inf], [inf] * 5)
    # a pilot run stopped at its grok carries the last value forward in the AUC
    r = _fake_run("s4", "R", 0, [0.0] * 5 + [1.0] * 6)
    r2 = dict(r, curve=r["curve"][:6], stopped_at=501)
    assert abs(ts.auc(r2) - ts.auc(r)) < 1e-12 and abs(ts.auc(r) - 6 / 11) < 1e-12
    try:
        ts.auc(dict(r, curve=r["curve"][:6]))
    except AssertionError:
        pass
    else:
        raise SystemExit("an incomplete main-run curve must not get an AUC")
    print("  summary known answers pass (partial rescue -> PASS (AUC); floor -> uninformative; stopped-run AUC carry-forward)")


def _corrupt(mul: np.ndarray) -> np.ndarray:
    """A table that is no longer a group (two entries swapped in one row): the closure must derive a wrong cell."""
    m = mul.copy(); m[1, 2], m[1, 3] = m[1, 3], m[1, 2]
    return m


def smoke(out: Path) -> dict:
    timing = {}
    for task in ("s4", "dim"):
        for frac in (0.3, 0.6):
            d = out / f"smoke_{task}_{frac}"
            t = time.time()
            r = subprocess.run([PY, str(HERE / "toy_struct.py"), "--task", task, "--frac", str(frac), "--steps", "300",
                                "--seeds", "0", "--out", str(d), "--device", "cpu", "--threads", "1"],
                               check=True, capture_output=True, text=True)
            runs = {p.stem.split("_seed")[0][4:]: json.loads(p.read_text()) for p in d.glob("run_*_seed0.json")}
            assert set(runs) == set(ts.ARMS[task].split(",")), runs.keys()
            timing[(task, frac)] = float(np.mean([x["ms_per_step"] for x in runs.values()]))
            none = runs["none"]
            for arm, x in runs.items():
                c = x["curve"]
                assert [e["step"] for e in c] == [0, 100, 200, 299]
                assert all(np.isfinite(e["ce"]) and np.isfinite(e["test_ce"]) and 0 <= e["test"] <= 1 for e in c)
                Dt = ts.build_task(task)
                _, tr_, te_ = ts.split(Dt, frac, 0)
                assert x["closure"] == 1.0 and x["n_train"] == len(tr_) and x["split"] == ts.split_mode(task)
                assert x["n_fresh"] == int(ts.fresh_mask(Dt, tr_, te_).sum()) and all(e["test_fresh"] is not None for e in c)
                if task == "dim":   # class split: every held-out question is fresh
                    assert all(e["test_fresh"] == e["test"] for e in c)
                key = "coset_share" if task == "s4" else "drop_share"
                assert all(e[key] is None or 0 <= e[key] <= 1 for e in c)
                if arm == "none":
                    assert x["target"] is None and all(e["geo"] is None and not e["prior_on"] for e in c)
                    continue
                assert all(e["prior_on"] for e in c)
                own = f"cka_{arm}"
                assert c[-1][own] > none["curve"][-1][own] + 0.1 and c[-1][own] > 0.8, (task, frac, arm, c[-1][own], none["curve"][-1][own])
                print(f"  {task} {frac} {arm:>8}: CKA own {x['cka_init'][arm]:.2f} -> {c[-1][own]:.2f} (none {none['curve'][-1][own]:.2f}); "
                      f"train {c[-1]['train']:.2f} test {c[-1]['test']:.2f}; {key}@299 "
                      f"{'-' if c[-1][key] is None else round(c[-1][key], 2)} (n_wrong {c[-1]['n_wrong']})")
            print(f"  {task} {frac}: {timing[(task, frac)]:.2f} ms/step (CPU, 1 thread, eval every 100)")
            assert not list(d.glob(".tmp_*")), "temporary run files left behind"
            S = subprocess.run([PY, str(HERE / "toy_struct.py"), "--summarize", str(d)], check=True, capture_output=True, text=True)
            assert "rule (a)" in S.stdout and (d / "summary.json").exists()
    # the window switches off: --win 0,150 -> prior on at evals 0 and 100, off at 200 and 299; per-arm overrides
    d = out / "smoke_window"
    subprocess.run([PY, str(HERE / "toy_struct.py"), "--task", "s4", "--frac", "0.6", "--steps", "300", "--seeds", "0",
                    "--arms", "R,R@always,R@200-250", "--win", "0,150", "--out", str(d), "--device", "cpu", "--threads", "1"],
                   check=True, capture_output=True)
    on = {a: [e["prior_on"] for e in json.loads((d / f"run_{a}_seed0.json").read_text())["curve"]] for a in ("R", "R@always", "R@200-250")}
    assert on == {"R": [True, True, False, False], "R@always": [True] * 4, "R@200-250": [False, False, True, False]}, on
    assert json.loads((d / "run_R@always_seed0.json").read_text())["target"] == "R"
    c = json.loads((d / "run_R_seed0.json").read_text())["curve"]
    print("  window [0, 150): prior_on at evals 0/100/200/299 =", on["R"], "| R@always", on["R@always"], "| R@200-250", on["R@200-250"])
    # paired init: every arm of a seed starts from the same weights (CKA at init identical across arms)
    for task in ("s4", "dim"):
        runs = [json.loads(p.read_text()) for p in (out / f"smoke_{task}_0.3").glob("run_*_seed0.json")]
        inits = {json.dumps(r["cka_init"], sort_keys=True) for r in runs}
        assert len(inits) == 1, "arms of one seed do not share the init"
    print("  all arms of a seed share the init (identical cka_init)")
    return timing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_known", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("1. known answer (mod vs toy_grok.py, 200 steps)")
    if not args.skip_known:
        known_answer(out)
    print("2. structure asserts")
    structure()
    print("2b. summary / decision-rule known answers (synthetic cells)")
    summary_known_answers(out)
    print("3. smoke (300 steps, every arm, CPU)")
    timing = smoke(out)
    print("4. timing (CPU, 1 thread):", {f"{k[0]}@{k[1]}": round(v, 2) for k, v in timing.items()})
    (out / "timing_cpu.json").write_text(json.dumps({f"{k[0]}@{k[1]}": v for k, v in timing.items()}))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
