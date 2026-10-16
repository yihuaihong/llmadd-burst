"""Tensor-level digest of a HF safetensors checkpoint AS LOADED IN BF16 (what ft36 / manifold_train load with
torch_dtype=bfloat16): sha256 over (name, shape, bf16 bytes) of every tensor, in sorted name order; floating tensors are
cast to bf16, others hashed as stored. Torch keeps the raw fp32 HF snapshots and Burst the bf16 conversions of
fetch_olmo.py, so file hashes cannot match across the sites; this digest is equal iff the loaded weights are identical
(2026-10-16, Burst infra; Torch reference digests in report/audits/burst_infra_AUDIT.md). One tensor at a time, so
peak RAM stays near the largest tensor.

    python tools/hash_ckpt.py <checkpoint dir> [...] [--expect <digest>,<digest>,...]
prints "<first 16 hex>  <n> tensors  <dir>" per dir; with --expect (one 16-hex digest per dir, in order) exits 3 on
any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def digest(d) -> tuple[str, int]:
    names = {}
    for shard in sorted(Path(d).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                names[k] = shard
    assert names, f"no *.safetensors in {d}"
    h = hashlib.sha256()
    open_files = {}
    for k in sorted(names):
        sh = names[k]
        if sh not in open_files:
            open_files.clear()
            open_files[sh] = safe_open(str(sh), framework="pt")
        t = open_files[sh].get_tensor(k)
        if t.is_floating_point():
            t = t.to(torch.bfloat16)
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.contiguous().view(torch.uint8).numpy() if t.dtype == torch.bfloat16 else t.contiguous().numpy())
    return h.hexdigest()[:16], len(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--expect", default=None, help="comma-separated 16-hex digests, one per dir; exit 3 on any mismatch")
    args = ap.parse_args()
    expect = args.expect.split(",") if args.expect else None
    assert expect is None or len(expect) == len(args.dirs), "--expect needs one digest per dir"
    bad = 0
    for i, d in enumerate(args.dirs):
        try:
            dg, n = digest(d)
        except (AssertionError, OSError) as e:   # a missing or unreadable checkpoint: report it, go on with the others
            print(f"MISSING  {d}: {e}", flush=True); bad += 1
            continue
        ok = "" if expect is None else ("  OK" if dg == expect[i] else f"  MISMATCH (expected {expect[i]})")
        bad += bool(expect is not None and dg != expect[i])
        print(f"{dg}  {n} tensors  {d}{ok}", flush=True)
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
