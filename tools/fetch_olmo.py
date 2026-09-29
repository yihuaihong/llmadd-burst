"""Download one OLMo-2 revision (any size, --repo) and store it as bf16 safetensors.

The Hub copy is fp32 (~29 GB). Shards are converted one tensor at a time, so peak RAM stays near one
bf16 shard. The fp32 download is deleted afterwards. A `.complete` marker makes reruns no-ops.

    python tools/fetch_olmo.py --revision main --dest /scratch/$USER/models/OLMo-2-1124-7B/main
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ID = "allenai/OLMo-2-1124-7B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repo", default=REPO_ID, help="any OLMo-2 size, e.g. allenai/OLMo-2-0425-1B, allenai/OLMo-2-1124-13B")
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    dest = Path(args.dest)
    if (dest / ".complete").exists():
        print(f"{args.revision}: already present at {dest}")
        return
    tmp = dest.parent / f".{dest.name}.fp32"
    shutil.rmtree(tmp, ignore_errors=True)   # leftovers of an interrupted run
    snapshot_download(args.repo, revision=args.revision, local_dir=tmp, max_workers=8)
    dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for shard in sorted(tmp.glob("*.safetensors")):
        tensors = {}
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                tensors[name] = t.to(torch.bfloat16) if t.is_floating_point() else t
                total += tensors[name].numel() * tensors[name].element_size()
        save_file(tensors, dest / shard.name, metadata={"format": "pt"})
        print(f"{args.revision}: converted {shard.name}")
        del tensors, t
        gc.collect()

    for path in tmp.iterdir():
        if path.suffix == ".safetensors" or path.is_dir():
            continue
        shutil.copy2(path, dest / path.name)
    index = dest / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        data.setdefault("metadata", {})["total_size"] = total
        index.write_text(json.dumps(data, indent=2))
    config = dest / "config.json"
    cfg = json.loads(config.read_text())
    cfg["torch_dtype"] = "bfloat16"
    config.write_text(json.dumps(cfg, indent=2))

    (dest / ".complete").write_text(args.revision + "\n")
    # Best effort: on NFS, files still mapped by this process linger as .nfsXXXX until it exits, so the
    # directory may not be removable yet; the next run clears it.
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{args.revision}: done, {total / 1e9:.1f} GB bf16 at {dest}")


if __name__ == "__main__":
    main()
