"""Download one OLMo-2 revision and store it as bf16 safetensors.

The Hub copy is fp32 (~29 GB). Shards are converted one tensor at a time, so peak RAM stays near one
bf16 shard. The fp32 download is deleted afterwards. A `.complete` marker makes reruns no-ops.

    python tools/fetch_olmo.py --revision main --dest /scratch/$USER/models/OLMo-2-1124-7B/main
"""

import argparse
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
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    dest = Path(args.dest)
    if (dest / ".complete").exists():
        print(f"{args.revision}: already present at {dest}")
        return
    tmp = dest.parent / f".{dest.name}.fp32"
    snapshot_download(REPO_ID, revision=args.revision, local_dir=tmp, max_workers=8)
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
        del tensors

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

    shutil.rmtree(tmp)
    (dest / ".complete").write_text(args.revision + "\n")
    print(f"{args.revision}: done, {total / 1e9:.1f} GB bf16 at {dest}")


if __name__ == "__main__":
    main()
