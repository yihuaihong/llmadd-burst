"""Evaluate saved from-scratch models (pretrain_helix.py --save_model) on every eval set of arithmetic_sets(), including
the strict subset (held-out pairs whose swap is not trained either). Same sets and context chunks as during training.

    python tools/pt_eval_saved.py --out <json> <run dir with model_*.pt and run_*.json> [...]
"""

from __future__ import annotations

import argparse
import glob
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from pretrain_helix import arithmetic_sets, build_model


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res, cache = {}, {}
    for d in a.dirs:
        for mp in sorted(glob.glob(f"{d}/model_*.pt")):
            rp = Path(mp).with_name(Path(mp).name.replace("model_", "run_").replace(".pt", ".json"))
            args = Namespace(**json.loads(rp.read_text())["args"])
            args.ctx_eval = True
            key = (args.tokenizer, args.data, args.seq)
            if key not in cache:
                tok = AutoTokenizer.from_pretrained(args.tokenizer)
                val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint32, mode="r")
                cache[key] = (tok, arithmetic_sets(args, tok, val, dev)[3])
            tok, evalsets = cache[key]
            model = build_model(args, tok, dev)
            model.load_state_dict(torch.load(mp, map_location=dev)); model.eval()
            r = {}
            with torch.autocast(dev, dtype=torch.bfloat16):
                for name, (ids, ans) in evalsets.items():
                    hit = [model(input_ids=ids[i:i + 250], logits_to_keep=1).logits[:, -1].argmax(-1) == ans[i:i + 250]
                           for i in range(0, len(ids), 250)]
                    r[name] = float(torch.cat(hit).float().mean())
            r["n"] = {k: int(v[0].shape[0]) for k, v in evalsets.items()}
            res[f"{Path(d).name}/{Path(mp).stem}"] = dict(arm=args.arm, seed=args.seed, window=args.window,
                                                          geo_site=getattr(args, "geo_site", "prefix"),
                                                          split=getattr(args, "split", "ordered"), **r)
            print(Path(mp).stem, {k: round(100 * v, 1) for k, v in r.items() if k != "n"}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
