"""Pre-registered pass rule of xsite53 (Batch 1 B1: does Burst A100 40GB reproduce Torch L40S?), exactly as in
report/audits/burst_infra_AUDIT.md 'Pass rule' (P0 / P1a / P1b, revised 2026-10-16 before any result), plus the measured
A100 hours per run and the factor that rescales every Burst time limit. Written and tested on fake results before any
xsite53 result existed; area by tools/readout_rules.py (= pt999_summary's formula), Burst re-runs as pt2main_summary.
  Burst: results/34[0-7]-xsite53-{none,win-ctx}-s<seed>/ (x53-7pct-<arm>/run_*.json, p0_check.json, offline_lm.json,
         dry_inputs.txt, PINNED.txt, gpu.csv) and status/<task>.json; P1b: the pt2b lr 1e-3 none / win-ctx tasks 350-353
         (and any EXTRA re-run of them, tools/make_pt2main_tasks.py) with their check.json
  Torch: runs/pt999/out/7pct-none (none), runs/ptrangeg/out/whole-helix3-w0.05-0.4 (W), runs/pt999/out/2pct-none and
         2pct-win-ctx (P1b; 2pct-win-ctx = Batch 1 T2), runs/ptgw/out/main-W (cka, secondary)
P0 exactness (all required):
  P0.1 task 332-data COMPLETED with rc 0 (it exits 3 on any tokenizer / parquet / bin / CPU dry-input mismatch)
  P0.2 per task: the sha256 of the 9 selected --dry_inputs lines = Torch's for the seed (recomputed here from
       dry_inputs.txt); PINNED.txt names a3bce3ab63c1; code_sha a3bce3ab63c1; last eval at step 18309; the run args = the
       Torch cell's except data / tokenizer / out / tag, keys new in the pinned code at their inert defaults
  P0.3 per seed and arm: |step-0 lm - Torch| <= 1e-3 and |step-0 val_loss - Torch| <= 2e-3
  P0.4 W: prior_events == [[916, 1], [7324, 0]] and geo > 0 at every eval with the prior on; none: never on
  P0.5 offline_lm on the full val, |orig_minus_logged| < 0.002
  (P0.2-5 are also the task's own p0_check.json: both paths must agree)
P1a, 7% (exactness near the ceiling): W area Burst - Torch, paired by seed: |mean diff| <= 1.5 and every |diff| <= 3.
     Sanity (not a rule): Burst W - none > 0 in 4/4 seeds.
P1b, 2% (required): pt2b-lr1e-3-none - Torch pt999 2pct-none, seeds 0-3: |mean diff| <= 1.9 and every |diff| <= 3.8;
     pt2b-lr1e-3-win-ctx - Torch pt999 2pct-win-ctx (T2), seeds 0-3: |mean diff| <= max(1.5, 1.07 SD) and every
     |diff| <= max(3, 2.13 SD), SD = T2's 8-seed area SD (ddof 1, as the 1.78 of 2pct-none). Burst runs: check.json
     passed and step-0 lm within 1e-3 of Torch's (a failed check withholds, a step-0 mismatch fails). 'incomplete' until
     the Burst cells and T2 exist. The Torch references must be valid (readout_rules.check_2pct, runs/ptlr/AUDIT.md
     'Controls': code_sha 79e9b410dcc3, 18310 steps / 75 evals, the same seed's 2pct-none args but target / schedule,
     step-0 lm vs the 2% values, T2 prior_on_frac 6408 / 18310, 2pct-none never on); an invalid one WITHHOLDS the rule.
Re-runs: the xsite53 run of (arm, seed) is the first FINISHED run in task order among results/<n>-xsite53-<arm>-s<seed>
(the 34x task, then any re-run under a new number <n>); an unfinished run (TIMEOUT: the tick does not retry it) or a
missing one keeps that seed INCOMPLETE until a re-run finishes. A re-run never replaces a finished run (P0 is the
site-exactness check: a P0 failure stands until its cause is found); later finished runs are listed and ignored.
Secondary (descriptive, not in the verdict): W b33_area |mean diff| <= 5; Burst mean W - none within 37.9 +- 12; offline
bpb |mean diff| <= 0.006 (Torch offline values via --torch_offline, else in-training bpb only); final cka_ops_helix3
|mean diff| <= 0.05 vs ptgw main-W once it exists.
Hours: per run projected_run_h (median sec/step between evals x steps), the last eval's sec, sacct-free elapsed_min of
the last attempt (status/), GPU utilisation median and peak memory (gpu.csv); factor = median projected_run_h / 0.75 h
(the 53M planning rate), which rescales every Burst 53M estimate and, until a 190M run is timed, the 190M rate
(2.0 A100-h per 3e8 tokens); each current pt2main / xsite time limit is printed as a multiple of its rescaled expected
time (the convention is 1.5-2x).
    python tools/xsite_summary.py [--results results] [--status status] [--torch /scratch/yh6210/llmadd_torch/runs]
                                  [--torch_offline f.json ...]
Exit 0 = P0, P1a, P1b-none and P1b-win-ctx all pass; 1 = a rule fails; 2 = something is incomplete or withheld (none fails).
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_pt2main_tasks as G   # noqa: E402  (pt2b task names, EXTRA re-runs, time limits)
import readout_rules as R        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(HERE.parent / "results"))
ap.add_argument("--status", default=str(HERE.parent / "status"))
ap.add_argument("--torch", default="/scratch/yh6210/llmadd_torch/runs")
ap.add_argument("--torch_offline", nargs="*", default=[], help="offline_lm.py outputs for the Torch reference runs")
A = ap.parse_args()
RES, STATUS, TORCH = Path(A.results), Path(A.status), Path(A.torch)

SEEDS = (0, 1, 2, 3)
PIN_SHA = "a3bce3ab63c1"
# sha256 of the 9 selected --dry_inputs lines, from logs/llmadd_pt999-7pct-none-s<seed>_*.out (= tasks/34x)
DRY = {0: "af96f9c50f36e2f18dfae7c55121a57180afb1fd34076fa3c0b3c2775738e1dc",
       1: "a261e3574bf6ecc9e1a1c920bcd5533db29549fe55170fdcc1345273b9040cf7",
       2: "52ab489ca66f606a304ec196fce9239a946bb9cd3af9a8b67f0085696ae6579c",
       3: "01b0d252ca1122a987d4e10a62ab0e72d54cef20863a75f79f393ef05a07cc98"}
DRY_RE = re.compile(rb"^(line:|held_b|ood_b|\{'held_b11'|model:|train row)")   # the tasks' grep -E, on bytes
ARMS = {"none": ("x53-7pct-none", "pt999/out/7pct-none", "none"),
        "W": ("x53-7pct-win-ctx", "ptrangeg/out/whole-helix3-w0.05-0.4", "win-ctx")}
TASK = {(arm, s): f"{340 + 2 * s + (arm == 'W')}-xsite53-{ARMS[arm][2]}-s{s}" for arm in ARMS for s in SEEDS}   # base tasks
# known answers (burst_infra_AUDIT.md reference table): the Torch areas the tolerances were set against
KNOWN = {"none": (59.90, 51.54, 52.29, 40.04), "W": (88.99, 88.38, 89.14, 88.92), "2pct-none": (13.07, 8.22, 9.68, 9.28)}
INERT = {"gate_start": 0.0, "gate_hyst": 0.0, "n_prefix": 256, "site": "ctx", "ckpt_every": 2000, "stop_after": -1,
         "text_min": 16, "grad_ckpt": "auto", "lm_chunk_tokens": -1, "untie": False}   # absent in the older Torch code
PATHS = {"data", "tokenizer", "out", "tag"}
W_EVENTS = [[916, 1], [7324, 0]]
PLAN_53M_H, PLAN_190M_H = 0.75, 2.0   # A100-h per 3e8 tokens (Batch 1 plan)

state = {}   # rule -> 'PASS' / 'FAIL' / 'INCOMPLETE' / 'WITHHELD'


def put(rule, st, txt):
    state[rule] = st
    print(f"{st:10s} {rule}: {txt}")


def one_run(d):
    fs = sorted(Path(d).glob("run_*.json"))
    assert len(fs) <= 1, fs
    return (fs[0], json.loads(fs[0].read_text())) if fs else (None, None)


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


# ---------------------------------------------------------------- read
torch = {}
for arm, (_, tdir, _) in ARMS.items():
    for s in SEEDS:
        f = next((TORCH / tdir).glob(f"run_*seed{s}_*.json"))
        torch[(arm, s)] = json.loads(f.read_text())
def task_names(arm, s):
    """the base task and its re-runs (results/ or status/ entries <n>-xsite53-<arm>-s<seed>), in task order"""
    suf = f"-xsite53-{ARMS[arm][2]}-s{s}"
    names = {TASK[(arm, s)]} | {d.name for d in RES.glob(f"*{suf}") if re.fullmatch(r"\d+" + re.escape(suf), d.name)} \
        | {f.stem for f in STATUS.glob(f"*{suf}.json") if re.fullmatch(r"\d+" + re.escape(suf), f.stem)}
    return sorted(names, key=lambda n: int(n.split("-")[0]))


burst, why_missing, rerun_notes = {}, {}, []
for (arm, s) in TASK:
    seen = []   # (name, state text) of the runs before the counted one
    for name in task_names(arm, s):
        d, st = RES / name, jload(STATUS / f"{name}.json")
        stt = f"status {st.get('state')}, rc {st.get('rc')}" if st else "no status"
        f, r = one_run(d / ARMS[arm][0]) if d.exists() else (None, None)
        if (arm, s) in burst:
            if r is not None and R.finished(r):
                rerun_notes.append(f"ignored {name}: a finished run of {ARMS[arm][2]} seed {s}; {burst[(arm, s)]['name']} counts "
                                   f"(the first finished run in task order; a re-run never replaces a finished run)")
            continue
        if r is None:
            seen.append(f"{name}: " + (f"no run JSON ({stt})" if d.exists() else f"not returned yet ({stt})"))
            continue
        if not R.finished(r):
            seen.append(f"{name}: unfinished, last eval {r['curve'][-1]['step']} of {r['steps']} ({stt})")
            continue
        burst[(arm, s)] = {"name": name, "dir": d, "file": f, "run": r, "p0": jload(d / "p0_check.json"),
                           "off": jload(d / "offline_lm.json"), "status": st}
        rerun_notes += [f"superseded: {x} -> {name} counts" for x in seen]
    if (arm, s) not in burst:
        why_missing[(arm, s)] = "; ".join(seen) + ("; a re-run (new task number, same name after it) counts once it "
                                                    "finishes" if any("unfinished" in x or "no run JSON" in x or "status" in x
                                                                      and "no status" not in x for x in seen) else "")

# known answers: the Torch reference areas are the ones the tolerances were computed from
for arm in ARMS:
    got = [round(R.area(torch[(arm, s)]), 2) for s in SEEDS]
    if any(abs(g - k) > 0.006 for g, k in zip(got, KNOWN[arm])):
        print(f"Torch reference areas of {arm} are {got}, not the AUDIT's {KNOWN[arm]}: the reference changed"); sys.exit(3)
print(f"(known answers: Torch areas none {KNOWN['none']}, W {KNOWN['W']} reproduced from the run JSONs)")

# ---------------------------------------------------------------- P0
print("== P0 exactness")
for v in why_missing.values():
    print("NOTE incomplete:", v)
for v in rerun_notes:
    print("NOTE", v)
st332 = jload(STATUS / "332-data.json")
if st332 is None:
    put("P0.1", "INCOMPLETE", "status/332-data.json missing")
else:
    ok = st332.get("state") == "COMPLETED" and str(st332.get("rc")) == "0"
    put("P0.1", "PASS" if ok else "FAIL", f"332-data {st332.get('state')} rc {st332.get('rc')}")
p0 = {k: [] for k in ("P0.2", "P0.3", "P0.4", "P0.5", "p0_check")}
for (arm, s), b in sorted(burst.items()):
    r, c, t = b["run"], b["run"]["curve"], torch[(arm, s)]
    name = b["name"]
    dry = b["dir"] / "dry_inputs.txt"
    if dry.exists():   # = grep -E ... | head -n 9 | sha256sum (lines split on \n only, each written back with \n)
        sel = [ln for ln in dry.read_bytes().split(b"\n") if DRY_RE.match(ln)][:9]
        dig = hashlib.sha256(b"".join(ln + b"\n" for ln in sel)).hexdigest()
        if dig != DRY[s]:
            p0["P0.2"].append(f"{name}: dry_inputs sha256 {dig[:12]} != Torch {DRY[s][:12]}")
    else:
        p0["P0.2"].append(f"{name}: dry_inputs.txt missing")
    pinned = (b["dir"] / "PINNED.txt").read_text() if (b["dir"] / "PINNED.txt").exists() else ""
    if not re.search(rf" {PIN_SHA}[0-9a-f]* tools/pretrain_range.py ", pinned):
        p0["P0.2"].append(f"{name}: PINNED.txt does not name {PIN_SHA} for pretrain_range.py")
    if r.get("code_sha") != PIN_SHA:
        p0["P0.2"].append(f"{name}: code_sha {r.get('code_sha')}")
    if not (c[-1]["step"] == r["steps"] - 1 == 18309):
        p0["P0.2"].append(f"{name}: last eval at step {c[-1]['step']} of {r['steps']}")
    ab, at = r["args"], t["args"]
    diff = {k: (ab.get(k), at.get(k)) for k in set(ab) | set(at) if k not in PATHS and k in ab and k in at and ab[k] != at[k]}
    new = {k: ab[k] for k in set(ab) - set(at) if k not in PATHS and INERT.get(k, object()) != ab[k]}
    if diff or new or set(at) - set(ab):
        p0["P0.2"].append(f"{name}: args differ from Torch {diff}, non-inert new keys {new}, missing {sorted(set(at) - set(ab))}")
    d_lm, d_val = c[0]["lm"] - t["curve"][0]["lm"], c[0]["val_loss"] - t["curve"][0]["val_loss"]
    b["d_lm"] = d_lm
    if not (abs(d_lm) <= 1e-3 and abs(d_val) <= 2e-3):
        p0["P0.3"].append(f"{name}: step-0 lm {d_lm:+.2e}, val_loss {d_val:+.2e} vs Torch")
    if arm == "W":
        on = [e for e in c if e["prior_on"]]
        if r.get("prior_events") != W_EVENTS or not on or not all((e["geo"] or 0) > 0 for e in on):
            p0["P0.4"].append(f"{name}: prior_events {r.get('prior_events')}, {len(on)} evals with the prior on, "
                              f"geo > 0 at {sum((e['geo'] or 0) > 0 for e in on)}")
    elif not (c[-1]["prior_on_frac"] == 0 and r.get("prior_events") == []):
        p0["P0.4"].append(f"{name}: prior on (on_frac {c[-1]['prior_on_frac']}, events {r.get('prior_events')})")
    o = (b["off"] or {}).get(str(Path(b["file"]).resolve()))
    o = o or next((v for v in (b["off"] or {}).values() if v.get("tag") == r["args"]["tag"] and v.get("seed") == s), None)
    b["bpb_off"] = o["bpb_full"] if o and o.get("max_windows") == 0 else None
    if not (o and o.get("max_windows") == 0 and abs(o.get("orig_minus_logged", math.inf)) < 0.002):
        p0["P0.5"].append(f"{name}: offline_lm " + ("missing" if not o else f"max_windows {o.get('max_windows')}, "
                                                                          f"orig_minus_logged {o.get('orig_minus_logged')}"))
    mine = not any(x.startswith(name + ":") for k in ("P0.2", "P0.3", "P0.4", "P0.5") for x in p0[k])
    theirs = b["p0"] and b["p0"].get("p0_pass")
    if b["p0"] is None:
        p0["p0_check"].append(f"{name}: p0_check.json missing")
    elif bool(theirs) != mine:
        p0["p0_check"].append(f"{name}: the task's p0_check says {'PASS' if theirs else 'FAIL'}, recomputed here "
                              f"{'PASS' if mine else 'FAIL'} ({[k['check'] for k in b['p0']['checks'] if not k['ok']]})")
    elif abs(b["p0"]["info"]["area"] - R.area(r)) > 0.006:
        p0["p0_check"].append(f"{name}: p0_check area {b['p0']['info']['area']} vs {R.area(r):.2f} here")
desc = {"P0.2": "dry inputs, PINNED, code_sha, finished, args", "P0.3": "step-0 lm / val_loss vs Torch",
        "P0.4": "prior on / off as designed", "P0.5": "offline_lm full val, double path",
        "p0_check": "the task's own p0_check.json agrees with this recomputation"}
for k, msgs in p0.items():
    if msgs:
        put(k, "FAIL", f"{desc[k]}: " + "; ".join(msgs))
    elif len(burst) < len(TASK):
        put(k, "INCOMPLETE", f"{desc[k]}: {len(burst)}/{len(TASK)} tasks returned")
    else:
        put(k, "PASS", f"{desc[k]} ({len(burst)} tasks)" + (
            f"; |step-0 lm diff| max {max(abs(b['d_lm']) for b in burst.values()):.1e}" if k == "P0.3" else ""))


# ---------------------------------------------------------------- P1a
def tol_test(rule, x, y, tm, te, what):
    d = np.asarray(x) - np.asarray(y)
    ok = abs(d.mean()) <= tm and all(abs(d) <= te)
    put(rule, "PASS" if ok else "FAIL", f"{what}: Burst - Torch per seed {', '.join(f'{v:+.2f}' for v in d)}; mean "
                                        f"{d.mean():+.2f} (|.| <= {tm:.2f}), max |diff| {abs(d).max():.2f} (<= {te:.2f})")
    return d


print("== P1a, 7% (W at its ceiling: an exactness check, not a test of the 2% regime)")
have = {arm: all((arm, s) in burst for s in SEEDS) for arm in ARMS}
if have["W"]:
    tol_test("P1a", [R.area(burst[("W", s)]["run"]) for s in SEEDS], [R.area(torch[("W", s)]) for s in SEEDS], 1.5, 3.0,
             "W area")
else:
    put("P1a", "INCOMPLETE", f"W: {sum(('W', s) in burst for s in SEEDS)}/4 Burst runs")
if have["W"] and have["none"]:
    gap = np.array([R.area(burst[("W", s)]["run"]) - R.area(burst[("none", s)]["run"]) for s in SEEDS])
    print(f"{'SANITY ' + ('PASS' if (gap > 0).all() else 'FAIL'):10s} Burst W - none > 0 in 4/4 (not a rule; Torch 29-49): "
          f"{', '.join(f'{v:+.1f}' for v in gap)}" + ("" if (gap > 0).all() else "  -> investigate before using any result"))

# ---------------------------------------------------------------- P1b
print("== P1b, 2% (pt2b lr 1e-3, tasks 350-353, vs Torch pt999 2pct-none / 2pct-win-ctx)")
T = [(name, tag, s) for name, tag, _, _, block, arm, lr, s in G.tasks()
     if block == "pt2b" and lr == "1e-3" and arm in ("none", "win-ctx") and s in SEEDS]
picked, problems = R.pick_burst_runs(RES, T)
for p in problems:
    print("NOTE", p)
t2 = {}
for cell in ("2pct-none", "2pct-win-ctx"):
    for f in sorted((TORCH / "pt999/out" / cell).glob("run_*.json")):
        r = json.loads(f.read_text())
        if R.finished(r):
            t2[(cell, r["args"]["seed"])] = r
got = [round(R.area(t2[("2pct-none", s)]), 2) for s in SEEDS if ("2pct-none", s) in t2]
if len(got) == 4 and any(abs(g - k) > 0.006 for g, k in zip(got, KNOWN["2pct-none"])):
    print(f"Torch 2pct-none areas are {got}, not the AUDIT's {KNOWN['2pct-none']}: the reference changed"); sys.exit(3)
sd_none = np.std([R.area(t2[("2pct-none", s)]) for s in range(8)], ddof=1) if all(("2pct-none", s) in t2 for s in range(8)) else None
if sd_none is not None:
    print(f"(the tolerance basis: Torch 2pct-none 8-seed area SD {sd_none:.3f} -> 0.64 x SD = {0.64 * sd_none:.2f}; frozen "
          f"tolerances 1.9 / 3.8)")
for arm, tcell, rule in (("none", "2pct-none", "P1b-none"), ("win-ctx", "2pct-win-ctx", "P1b-win-ctx")):
    tag = f"pt2b-lr1e-3-{arm}"
    tseeds = range(8) if arm == "win-ctx" else SEEDS
    miss_b = [s for s in SEEDS if (tag, s) not in picked]
    miss_t = [s for s in tseeds if (tcell, s) not in t2 or ("2pct-none", s) not in t2]
    if miss_b or miss_t:
        put(rule, "INCOMPLETE", "; ".join(([f"Burst {tag} seeds {miss_b} not returned / unfinished"] if miss_b else [])
                                          + ([f"Torch {tcell} (or its 2pct-none pair) seeds {miss_t} missing / unfinished"
                                              + (" (Batch 1 T2)" if arm == "win-ctx" else "")] if miss_t else [])))
        continue
    inval = [f"Torch {tcell} seed {s}: {why}" for s in tseeds
             for why in R.check_2pct(t2[(tcell, s)], arm, t2[("2pct-none", s)]["args"])]
    if inval:
        put(rule, "WITHHELD", f"invalid Torch reference: {inval[0]}" + (f" (+{len(inval) - 1} more)" if len(inval) > 1 else ""))
        continue
    bad, lm = [], []
    for s in SEEDS:
        task, r = picked[(tag, s)]
        ck = jload(task / "check.json")
        dl = r["curve"][0]["lm"] - t2[(tcell, s)]["curve"][0]["lm"]
        lm.append(dl)
        if ck and abs(ck["info"]["area"] - R.area(r)) > 0.006:
            print(f"double path: check.json area {ck['info']['area']} vs {R.area(r):.2f} here ({task.name})"); sys.exit(3)
        if not (ck and ck.get("pass")):
            bad.append(f"check.json {'missing' if not ck else 'FAILED'} in {task.name}")
    if any(abs(v) > 1e-3 for v in lm):
        put(rule, "FAIL", f"step-0 lm vs Torch {tcell}: {', '.join(f'{v:+.1e}' for v in lm)} (pairing broken)")
        continue
    if bad:
        put(rule, "WITHHELD", "; ".join(bad))
        continue
    x = [R.area(picked[(tag, s)][1]) for s in SEEDS]
    y = [R.area(t2[(tcell, s)]) for s in SEEDS]
    if arm == "none":
        tol_test(rule, x, y, 1.9, 3.8, "none area")
    else:
        sd = np.std([R.area(t2[(tcell, s)]) for s in range(8)], ddof=1)
        tol_test(rule, x, y, max(1.5, 1.07 * sd), max(3.0, 2.13 * sd), f"win-ctx area (T2 8-seed SD {sd:.2f})")

# ---------------------------------------------------------------- secondary (descriptive)
print("== secondary (descriptive, not in the verdict)")
if have["W"]:
    d = np.array([R.area(burst[("W", s)]["run"], ("b33",)) - R.area(torch[("W", s)], ("b33",)) for s in SEEDS])
    print(f"  W b33_area Burst - Torch: mean {d.mean():+.2f} (<= 5: {'yes' if abs(d.mean()) <= 5 else 'NO'}); per seed "
          f"{', '.join(f'{v:+.1f}' for v in d)}")
if have["W"] and have["none"]:
    g = np.mean([R.area(burst[("W", s)]["run"]) - R.area(burst[("none", s)]["run"]) for s in SEEDS])
    print(f"  Burst mean W - none {g:+.1f} (Torch 37.9 +- 12: {'within' if abs(g - 37.9) <= 12 else 'OUTSIDE'}); none "
          f"Burst - Torch mean {np.mean([R.area(burst[('none', s)]['run']) - R.area(torch[('none', s)]) for s in SEEDS]):+.2f} "
          f"(none's seed SD 8.4: level not testable)")
toff = {}
for f in A.torch_offline:
    toff.update(jload(f) or {})
for arm in ARMS:
    if not have[arm]:
        continue
    db = [burst[(arm, s)]["run"]["curve"][-1]["val_bits_per_byte"] - torch[(arm, s)]["curve"][-1]["val_bits_per_byte"] for s in SEEDS]
    tf = {s: next((v for v in toff.values() if v.get("tag") == torch[(arm, s)]["args"]["tag"] and v.get("seed") == s
                   and v.get("max_windows") == 0), None) for s in SEEDS}
    txt = f"  {arm} bpb Burst - Torch: in-training (65k tokens) mean {np.mean(db):+.4f}"
    if all(tf.values()) and all(burst[(arm, s)]["bpb_off"] is not None for s in SEEDS):
        do = [burst[(arm, s)]["bpb_off"] - tf[s]["bpb_full"] for s in SEEDS]
        txt += f"; offline (full val) mean {np.mean(do):+.4f} (<= 0.006: {'yes' if abs(np.mean(do)) <= 0.006 else 'NO'})"
    else:
        txt += "; offline: no Torch offline bpb for every seed (pass --torch_offline)"
    print(txt)
mw = {}
for f in sorted((TORCH / "ptgw/out/main-W").glob("run_*.json")):
    r = json.loads(f.read_text())
    if R.finished(r):
        mw[r["args"]["seed"]] = r
if have["W"] and all(s in mw for s in SEEDS):
    d = np.mean([burst[("W", s)]["run"]["curve"][-1]["cka_ops_helix3"] - mw[s]["curve"][-1]["cka_ops_helix3"] for s in SEEDS])
    print(f"  W final cka_ops_helix3 Burst - ptgw main-W: mean {d:+.3f} (<= 0.05: {'yes' if abs(d) <= 0.05 else 'NO'})")
else:
    print(f"  W final cka_ops_helix3: ptgw main-W has {len(mw)}/4 finished seeds (ptrangeg's older code does not log it)")

# ---------------------------------------------------------------- hours
print("== measured A100 hours (53M, 3e8 tokens) and the rescaling factor")
proj, utils = [], []
for (arm, s), b in sorted(burst.items()):
    c = b["run"]["curve"]
    rate = [(q["sec"] - p["sec"]) / (q["step"] - p["step"]) for p, q in zip(c, c[1:]) if q["sec"] > p["sec"] and q["step"] > p["step"]]
    ph = statistics.median(rate) * b["run"]["steps"] / 3600 if rate else None
    if b["p0"] and b["p0"]["info"].get("projected_run_h") is not None and ph is not None \
            and abs(b["p0"]["info"]["projected_run_h"] - ph) > 1e-6:
        print(f"double path: projected_run_h {b['p0']['info']['projected_run_h']} in p0_check vs {ph} here"); sys.exit(3)
    util, mem = [], []
    gp = b["dir"] / "gpu.csv"
    for row in csv.reader(open(gp)) if gp.exists() else []:
        try:
            util.append(int(row[2].split()[0])); mem.append(int(row[4].split()[0]))
        except (IndexError, ValueError):
            pass
    st = b["status"] or {}
    if ph is not None:
        proj.append(ph)
    if util:
        utils.append(statistics.median(util))
    print(f"  {b['name']:26s} projected_run_h {ph if ph is None else round(ph, 3)}  last eval sec {c[-1]['sec'] / 3600:.3f} h  "
          f"elapsed (last attempt) {st.get('elapsed_min', '?')} min  GPU util median {statistics.median(util) if util else '?'}%  "
          f"peak {max(mem) if mem else '?'} MiB  ({len(util)} samples)")
if proj:
    m = statistics.median(proj)
    f = m / PLAN_53M_H
    print(f"  median projected {m:.3f} A100-h per 53M 3e8-token run -> factor {f:.2f} (measured / {PLAN_53M_H} h planned); "
          f"rescaled: 53M 3e8 {m:.2f} h, 53M 1.2e9 {4 * m:.2f} h, 190M 3e8 {PLAN_190M_H * f:.2f} h (190M not timed)")
    exp = {"pt2b": m, "pt2long": 4 * m, "pt2s190": PLAN_190M_H * f}
    for base, block, size, data, tokens, plan_h, tlim, _ in G.BLOCKS:
        x = G.hours(tlim) / exp[block]
        print(f"    {block:8s} limit {tlim} = {x:.1f} x the rescaled time" + ("  -> below 1.5x: raise it before these tasks are "
              "submitted (regenerate)" if x < 1.5 else "  (above 3x: could be lowered; costs only guard reservation)" if x > 3 else ""))
    x = 2.0 / m
    print(f"    xsite53  limit 02:00:00 = {x:.1f} x the measured time")
    if utils:
        print(f"  GPU utilisation: median over tasks {statistics.median(utils):.0f}%" + (
            " -> below 50%: 53M Burst tasks run 2 jobs per GPU (AUDIT rescaling rule)" if statistics.median(utils) < 50 else ""))

# ---------------------------------------------------------------- verdict
print("== verdict")
P0 = [state[k] for k in state if k.startswith("P0") or k == "p0_check"]
core = P0 + [state.get("P1a", "INCOMPLETE"), state.get("P1b-none", "INCOMPLETE")]
if "FAIL" in P0:
    print("OUTCOME: P0 FAILS: no Burst pretraining result is used until the cause is found")
elif "FAIL" in core:
    print(("OUTCOME: P0 passes, P1 FAILS" if all(v == "PASS" for v in P0) else "OUTCOME: P1 FAILS (P0 not complete yet)")
          + ": Burst only for self-contained comparisons (pt2main is designed that way); report the site offset")
elif all(v == "PASS" for v in core):
    w = state.get("P1b-win-ctx", "INCOMPLETE")
    print("OUTCOME: P0 + P1a + P1b-none pass: Burst cells may be pooled with Torch only as within-site paired differences "
          "(arm minus its own-site reference, site as a block); absolute levels per site; the arms of one comparison never "
          "on two sites. " + {"PASS": "P1b-win-ctx passes: the 2% pooling stands.",
                               "FAIL": "P1b-win-ctx FAILS: the 2% pooling is withdrawn."}.get(w, f"At 2% provisional until "
                                                                                                 f"P1b-win-ctx is in ({w})."))
else:
    print("OUTCOME: incomplete (" + ", ".join(f"{k} {v}" for k, v in state.items() if v != "PASS") + ")")
allv = list(state.values())
print("OVERALL", "FAIL" if "FAIL" in allv else "PASS" if all(v == "PASS" for v in allv) and "P1b-win-ctx" in state else "INCOMPLETE")
sys.exit(1 if "FAIL" in allv else 0 if all(v == "PASS" for v in allv) and "P1b-win-ctx" in state else 2)
