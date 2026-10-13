"""Target geometries read out of a trained model (automatic targets instead of a hand-designed helix): the states of
"Q: x", x = 10..99, saved by manifold_atlas.py (states_<tag>.npz; z-scored per dimension over all atlas items, then
centred over the 90 numbers), reduced to their top-k principal components. Writes <out>.npz with
  feat_k{k}   90 x k principal-component coordinates (rows = x = 10..99)
  feat_full   90 x d centred states (its Gram is the teacher's own Gram)
and prints how helix-like the teacher geometry is (linear CKA to the hand helix).

    python tools/teacher_targets.py --states runs/atlas/out/states_main.npz --layer 16 --out <file>
"""

from __future__ import annotations

import argparse

import numpy as np

import numlib as nl
from manifold_domains import domains


def cka_np(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    G, K = A @ A.T, B @ B.T
    return float((G * K).sum() / (np.linalg.norm(G) * np.linalg.norm(K)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--template", default="neutral", choices=("neutral", "cued"))
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.states)
    layers = list(z["layers"]); owner = z["owner"]
    D = domains()
    nums = [int(v) for v in D["numbers"]["items"]]
    assert nums == list(range(10, 100))
    H = z[a.template][layers.index(a.layer)][owner == "numbers"].astype(np.float64)
    H = H - H.mean(0)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    out = {"feat_full": H.astype(np.float32), "singular_values": S.astype(np.float32)}
    helix = nl.features(np.arange(10, 100))[0]
    print(f"teacher {a.states} L{a.layer} {a.template}: CKA(full, helix) = {cka_np(H, helix):.3f}; "
          f"variance in top 4 / 8 / 16 PCs = {S[:4].__pow__(2).sum() / (S ** 2).sum():.2f} / "
          f"{S[:8].__pow__(2).sum() / (S ** 2).sum():.2f} / {S[:16].__pow__(2).sum() / (S ** 2).sum():.2f}")
    for k in (int(v) for v in a.ks.split(",")):
        F = U[:, :k] * S[:k]
        out[f"feat_k{k}"] = F.astype(np.float32)
        print(f"  k={k}: CKA(feat_k{k}, helix) = {cka_np(F, helix):.3f}, CKA(feat_k{k}, full) = {cka_np(F, H):.3f}")
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
