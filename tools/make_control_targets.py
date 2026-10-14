"""Control targets for the teacher-target experiment (F77 audit, 2026-10-14), written in the --target_file format of
pretrain_helix.py (90 x k features for x = 10..99):
  helix_noisy  the hand helix plus Gaussian noise, noise level set so that the target AS TRAINED (columns z-scored)
               has linear CKA ~0.63 to the hand helix, matching the teacher's main-L16 k8 target (0.625)
  digit_onehot units one-hot (10) + tens one-hot (9): a hand-made number code that is not a helix
Prints the CKA of each (as trained) to the hand helix and to the teacher target.

    python tools/make_control_targets.py --out <dir> [--teacher <teacher npz>]
"""
import argparse
import numpy as np
import numlib as nl
from teacher_targets import cka_np, zcols

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--teacher", default=None)
ap.add_argument("--goal", type=float, default=0.625)
a = ap.parse_args()
x = np.arange(10, 100)
H = nl.features(x)[0]; hz = zcols(H)
rng = np.random.default_rng(0)
E = rng.normal(size=hz.shape)   # noise ON the helix columns (extra columns would get unit weight after z-scoring)
best = None
for s in np.linspace(0.0, 3.0, 601):   # noise sd relative to the helix's per-column sd (1 after z-scoring)
    F = hz + s * E
    c = cka_np(zcols(F), hz)
    if best is None or abs(c - a.goal) < abs(best[1] - a.goal):
        best = (s, c, F)
s, c, Fn = best
oh = np.concatenate([np.eye(10)[x % 10], np.eye(9)[x // 10 - 1]], 1)
np.savez(f"{a.out}/control_helix_noisy.npz", feat=Fn.astype(np.float32))
np.savez(f"{a.out}/control_digit_onehot.npz", feat=oh.astype(np.float32))
print(f"helix_noisy: noise scale {s:.2f}, CKA(as trained, helix) {c:.3f}, columns {Fn.shape[1]}")
print(f"digit_onehot: CKA(as trained, helix) {cka_np(zcols(oh), hz):.3f}, columns {oh.shape[1]}")
if a.teacher:
    T = zcols(np.load(a.teacher)["feat_k8"].astype(np.float64))
    print(f"teacher k8 vs helix {cka_np(T, hz):.3f}; vs helix_noisy {cka_np(T, zcols(Fn)):.3f}; vs digit_onehot {cka_np(T, zcols(oh)):.3f}")
