"""Offline LM re-evaluation of saved from-scratch models (tools/pretrain_range.py / pretrain_helix.py with --save_model).

The in-training LM metric (val_bits_per_byte) uses 8 x 16 fixed windows = 65,536 val tokens, too few to call a cost of
~0.004 bits/byte (F79). This script loads each saved model and measures bits per byte on ALL of val.bin in
non-overlapping 512-token windows (+1 target token; the same windows for every model), with the training-time eval
settings (eval mode, bf16 autocast). Double path: it also recomputes the in-training metric on the original windows
(np.random.default_rng(7), 8 batches x 16 offsets) and reports the difference to the run's last logged value.

    python tools/offline_lm.py --data <pretrain_data> --tokenizer <olmo2> --out <file.json> <run dir> [<run dir> ...]
(a run dir holds run_*.json and the matching model_*.pt)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from pretrain_helix import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_windows", type=int, default=0, help="testing only: limit the number of full-val windows")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint32, mode="r")
    res = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}

    def bpb(model, windows):
        """bits per byte and nats per token over windows [n, seq + 1] (targets = window[1:])."""
        nats, ntok, nbytes = 0.0, 0, 0
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for i in range(0, len(windows), args.bs):
                x = torch.tensor(windows[i:i + args.bs].astype(np.int64), device=dev)
                ce = Fn.cross_entropy(model(input_ids=x[:, :-1]).logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum")
                nats += float(ce); ntok += x[:, 1:].numel()
                nbytes += sum(len(tok.decode(row).encode("utf-8")) for row in x[:, 1:].tolist())
        return nats / math.log(2) / nbytes, nats / ntok, ntok

    for d in args.dirs:
        for f in sorted(glob.glob(f"{d}/run_*.json")):
            r = json.loads(Path(f).read_text())
            a = Namespace(**r["args"])
            key = str(Path(f).resolve())   # audit 2026-10-15: the run file itself (no basename collisions)
            old = res.get(key)
            if old and old.get("max_windows", 0) == 0 and old.get("bpb_logged_final") == r["curve"][-1]["val_bits_per_byte"]:
                continue   # reuse only a full-val entry computed from this very run (not a smoke test, not stale)
            mfile = Path(f).parent / f"model_{a.tokenization}_{a.target}_seed{a.seed}{a.tag}.pt"
            if not mfile.exists() or r["curve"][-1]["step"] < r["steps"] - 1:
                print("skip (no model or unfinished):", f, flush=True)
                continue
            model = build_model(a, tok, dev)
            model.load_state_dict(torch.load(mfile, map_location=dev))
            model.eval()
            seq = a.seq
            # original in-training windows (pretrain_range.evaluate): rng 7, 8 batches of 16 offsets
            vrr = np.random.default_rng(7)
            orig = np.stack([val[o:o + seq + 1] for _ in range(8) for o in vrr.integers(0, len(val) - seq - 1, size=16)])
            b_orig, _, _ = bpb(model, orig)
            nwin = (len(val) - 1) // seq
            if args.max_windows:
                nwin = min(nwin, args.max_windows)
            full = np.stack([val[i * seq: i * seq + seq + 1] for i in range(nwin)])
            b_full, npt_full, ntok = bpb(model, full)
            logged = r["curve"][-1]["val_bits_per_byte"]
            res[key] = {"run": f, "model": str(mfile), "tag": a.tag, "seed": a.seed, "code_sha": r.get("code_sha"),
                        "bpb_full": b_full, "nats_per_token_full": npt_full, "tokens_full": ntok,
                        "bpb_orig_windows": b_orig, "bpb_logged_final": logged, "orig_minus_logged": b_orig - logged,
                        "max_windows": args.max_windows}
            print(json.dumps({k: res[key][k] for k in ("tag", "seed", "bpb_full", "bpb_orig_windows", "orig_minus_logged")}), flush=True)
            Path(args.out).write_text(json.dumps(res, indent=1))
            del model
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
