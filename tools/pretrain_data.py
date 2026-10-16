"""Pretraining data for the from-scratch helix-prior experiment: FineWeb-Edu (sample-10BT) tokenized with the OLMo-2
tokenizer into flat uint32 token files (documents separated by EOS): train.bin (the first --tokens tokens) and
val.bin (the next --val_tokens). Arithmetic is not mixed in here; pretrain_helix.py appends it per sequence.
2026-10-16 (Burst rebuild of the Torch data): --revision pins the HF dataset commit (the Torch builds of pretrain_data
and pretrain_data_3B downloaded fineweb-edu at 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, raw/.cache/huggingface/
download/*.metadata); files missing under raw/ are fetched with huggingface_hub (the hf CLI is not on PATH inside Burst
tasks); the tokens are streamed to disk instead of concatenated in memory (the 3e9-token build needed ~2 x 12 GB, a
Burst CPU node has 23 GB). The bytes written are unchanged: train.bin = tokens [0, --tokens), val.bin = the next
--val_tokens of the same stream.

    python tools/pretrain_data.py --tokenizer <olmo2 ckpt dir> --out /scratch/.../pretrain_data --tokens 400000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=400_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--files", default="sample/10BT/000_00000.parquet,sample/10BT/001_00000.parquet")
    ap.add_argument("--repo", default="HuggingFaceFW/fineweb-edu", help="HF dataset repo (e.g. HuggingFaceTB/finemath)")
    ap.add_argument("--revision", default=None,
                    help="HF dataset revision (commit) at which missing --files are downloaded; default: the repo's main "
                         "branch, as before. Files already under <out>/raw are used as they are")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    try:   # already built with the same request (e.g. a queued copy of the job after a local run): nothing to do
        m = json.loads((out / "meta.json").read_text())
        if (m["files"], m["repo"], m["train_tokens"], m["val_tokens"], m.get("revision")) == \
                (args.files, args.repo, args.tokens, args.val_tokens, args.revision):
            print("already built:", out); return
    except (OSError, KeyError, ValueError):
        pass
    raw = out / "raw"
    for f in args.files.split(","):
        if not (raw / f).exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(args.repo, f, repo_type="dataset", revision=args.revision, local_dir=raw)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    need = args.tokens + args.val_tokens
    parts = ((out / "train.bin.tmp", 0, args.tokens), (out / "val.bin.tmp", args.tokens, need))   # [lo, hi) of the stream
    fh = [open(p, "wb", buffering=1 << 24) for p, _, _ in parts]
    n = 0
    for f in sorted(str(raw / f) for f in args.files.split(",")):   # exactly the requested files, in order
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            ids = tok(batch.column(0).to_pylist(), add_special_tokens=False)["input_ids"]
            for x in ids:
                a = np.asarray(x + [eos], dtype=np.uint32)
                for h, (_, lo, hi) in zip(fh, parts):   # the part of this document's tokens [n, n + len(a)) in [lo, hi)
                    i, j = max(lo - n, 0), min(hi - n, len(a))
                    if i < j:
                        h.write(a[i:j].tobytes())
                n += len(a)
            print(f"{n / 1e6:.0f}M tokens", flush=True)
            if n >= need:
                break
        if n >= need:
            break
    for h in fh:
        h.close()
    assert n >= need, f"only {n} tokens"
    for (p, _, _), name in zip(parts, ("train.bin", "val.bin")):   # atomic: readers never see half files
        p.replace(out / name)
    (out / "meta.json").write_text(json.dumps({"tokenizer": args.tokenizer, "train_tokens": args.tokens, "val_tokens": args.val_tokens,
                                               "eos": eos, "vocab": len(tok), "files": args.files, "repo": args.repo,
                                               "revision": args.revision}, indent=1))
    print("done", n)


if __name__ == "__main__":
    main()
