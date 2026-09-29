"""Download one revision of a Hugging Face model repo as-is (no conversion), e.g. Pythia checkpoints.

Only the PyTorch .bin weights (fp16 for Pythia, the smallest copy), config and tokenizer files are fetched, so
a stray safetensors index cannot point from_pretrained at shards that were not downloaded. `.complete` makes
reruns no-ops.

    python tools/fetch_hf.py --repo EleutherAI/pythia-1.4b --revision step1000 --dest /scratch/$USER/models/pythia-1.4b/step1000
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--allow", default="config.json,tokenizer*,special_tokens_map.json,pytorch_model*")
    args = ap.parse_args()
    dest = Path(args.dest)
    if (dest / ".complete").exists():
        print(f"{args.repo}@{args.revision}: already present")
        return
    snapshot_download(args.repo, revision=args.revision, local_dir=dest, max_workers=8, allow_patterns=args.allow.split(","))
    (dest / ".complete").write_text(args.revision + "\n")
    print(f"{args.repo}@{args.revision}: done -> {dest}")


if __name__ == "__main__":
    main()
