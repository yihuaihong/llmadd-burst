"""Digit-split copy of the pretraining token files (pretrain_data.py output): every pure-digit OLMo-2 token ("47", "194",
1110 such tokens, 1-3 digits) is replaced by its single-digit tokens ("4", "7"), everything else is unchanged. The
OLMo-2 tokenizer never merges digits with letters or spaces, so this changes ONLY how numbers are tokenised (as in
Qwen2.5 / Llama-2, which split every digit). Writes <out>/{train,val}.bin (uint32, train truncated to the same token
count as the source) and meta.json with the expansion statistics.

    python tools/digit_split_data.py --src /scratch/.../pretrain_data --out /scratch/.../pretrain_data_digits --tokenizer <olmo2>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def split_table(tok):
    """id -> list of single-digit ids for pure-digit tokens; None otherwise."""
    single = {c: tok.convert_tokens_to_ids(c) for c in "0123456789"}
    assert all(tok.decode([v]) == c for c, v in single.items())
    table = {}
    for i in range(len(tok)):
        s = tok.decode([i])
        if s and s.isascii() and s.isdigit():
            table[i] = [single[c] for c in s]
    return table


_LUT = {}


def expand(arr: np.ndarray, table: dict) -> np.ndarray:
    """Vectorised: a lookup table D[id, k] (k-th output token of id) and L[id] (number of output tokens)."""
    key = id(table)
    if key not in _LUT:
        V = max(max(table) + 1, 200_000)
        D = np.repeat(np.arange(V, dtype=np.int64)[:, None], 3, 1)
        L = np.ones(V, dtype=np.int64)
        for i, digs in table.items():
            L[i] = len(digs)
            D[i, :len(digs)] = digs
        _LUT[key] = (D, L)
    D, L = _LUT[key]
    arr = np.asarray(arr, dtype=np.int64)
    lens = L[arr]
    starts = np.cumsum(lens) - lens
    out = np.empty(int(lens.sum()), dtype=np.int64)
    for k in range(3):
        m = lens > k
        out[starts[m] + k] = D[arr[m], k]
    return out.astype(np.uint32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    table = split_table(tok)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {"source": args.src, "digit_tokens": len(table)}
    for name in ("train", "val"):
        src = np.memmap(Path(args.src) / f"{name}.bin", dtype=np.uint32, mode="r")
        n_target = len(src)
        chunks, total, used = [], 0, 0
        step = 20_000_000
        while total < n_target and used < len(src):
            piece = expand(np.asarray(src[used:used + step]), table)
            chunks.append(piece); total += len(piece); used += min(step, len(src) - used)
        res = np.concatenate(chunks)[:n_target]
        res.tofile(out / f"{name}.bin")
        sample = np.asarray(src[:200000])
        meta[name] = {"tokens": int(len(res)), "source_tokens_used": int(used),
                      "digit_token_share_source": float(np.isin(sample, list(table)).mean()),
                      "expansion": float(len(expand(sample, table)) / len(sample))}
        # check: decoding the split text equals decoding the source text on a sample
        a, b = tok.decode(sample[:5000].tolist()), tok.decode(expand(sample[:5000], table).tolist())
        assert a == b, "digit splitting changed the text"
        print(name, meta[name], flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
