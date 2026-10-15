"""Pretraining data for the from-scratch helix-prior experiment: FineWeb-Edu (sample-10BT) tokenized with the OLMo-2
tokenizer into flat uint32 token files (documents separated by EOS): train.bin (the first --tokens tokens) and
val.bin (the next --val_tokens). Arithmetic is not mixed in here; pretrain_helix.py appends it per sequence.

    python tools/pretrain_data.py --tokenizer <olmo2 ckpt dir> --out /scratch/.../pretrain_data --tokens 400000000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
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
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    try:   # already built with the same request (e.g. a queued copy of the job after a local run): nothing to do
        m = json.loads((out / "meta.json").read_text())
        if (m["files"], m["repo"], m["train_tokens"], m["val_tokens"]) == (args.files, args.repo, args.tokens, args.val_tokens):
            print("already built:", out); return
    except (OSError, KeyError, ValueError):
        pass
    raw = out / "raw"
    for f in args.files.split(","):
        if not (raw / f).exists():
            subprocess.run(["hf", "download", args.repo, "--repo-type", "dataset", "--include", f,
                            "--local-dir", str(raw)], check=True)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    need = args.tokens + args.val_tokens
    buf, n = [], 0
    for f in sorted(str(raw / f) for f in args.files.split(",")):   # exactly the requested files, in order
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            ids = tok(batch.column(0).to_pylist(), add_special_tokens=False)["input_ids"]
            for x in ids:
                buf.append(np.asarray(x + [eos], dtype=np.uint32)); n += len(x) + 1
            print(f"{n / 1e6:.0f}M tokens", flush=True)
            if n >= need:
                break
        if n >= need:
            break
    allt = np.concatenate(buf)
    assert len(allt) >= need, f"only {len(allt)} tokens"
    for name, part in (("train.bin", allt[: args.tokens]), ("val.bin", allt[args.tokens: need])):   # atomic: readers
        part.tofile(out / (name + ".tmp")); (out / (name + ".tmp")).replace(out / name)              # never see half files
    (out / "meta.json").write_text(json.dumps({"tokenizer": args.tokenizer, "train_tokens": args.tokens, "val_tokens": args.val_tokens,
                                               "eos": eos, "vocab": len(tok), "files": args.files, "repo": args.repo}, indent=1))
    print("done", len(allt))


if __name__ == "__main__":
    main()
