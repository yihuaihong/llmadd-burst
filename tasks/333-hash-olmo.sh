# kind: cpu
# time: 01:00:00
# cpus: 8
# after: 331-env311
# Are Burst's OLMo-2 7B checkpoints Torch's? (2026-10-16, Batch 1 B0, for the LoRA fine-tuning batches; intent spec
# report/audits/burst_infra_AUDIT.md). Torch keeps the fp32 HF snapshots, Burst the bf16 conversions of fetch_olmo.py, so
# file hashes differ by construction: tools/hash_ckpt.py digests the weights as loaded in bf16. 42B and 210B against the
# Torch digests (tools/hash_ckpt.py on Torch, 2026-10-16); tokenizer.json of 9B against Torch's; the other 5 checkpoints
# of the ft36b batches (9B 21B 84B 630B 3896B; plan section 0, Burst models) must be complete: fetch_olmo.py's .complete
# marker names the revision and every shard of model.safetensors.index.json is there and not empty. Every checkpoint dir
# is listed with its shard count. Nothing waits for this task. Exit 3 = a digest differs or a checkpoint is missing.
set -uo pipefail
ROOT=${LLMADD_ROOT:-/scratch/$USER}
M=$MODELS/OLMo-2-1124-7B; R=$OUT/hash_olmo.txt; bad=0
{
  echo "## checkpoints in $M ($(date -u +%FT%TZ)): shards, .complete marker"
  for d in "$M"/*/; do echo "  $(ls "$d"*.safetensors 2>/dev/null | wc -l) $( [ -e "$d.complete" ] && echo complete || echo NOT-complete) $d"; done
  echo "## tokenizer.json of stage1-step2000-tokens9B (Torch 73fd5254...)"
} > "$R" 2>&1
(cd "$M/stage1-step2000-tokens9B" && sha256sum -c -) >> "$R" 2>&1 <<'SHA' || bad=1
73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca  tokenizer.json
SHA
echo "## bf16 tensor digests (Torch: 42B 4133fa61ec69ece6, 210B 08e4c3a401069fd0; 355 tensors each)" >> "$R"
PYTHONDONTWRITEBYTECODE=1 "$ROOT/llmadd_env311/bin/python" tools/hash_ckpt.py "$M/stage1-step10000-tokens42B" \
  "$M/stage1-step50000-tokens210B" --expect 4133fa61ec69ece6,08e4c3a401069fd0 >> "$R" 2>&1 || bad=1
echo "## the other ft36b checkpoints: .complete names the revision, every indexed shard present and not empty" >> "$R"
"$ROOT/llmadd_env311/bin/python" - "$M" >> "$R" 2>&1 <<'PY' || bad=1
import json, sys
from pathlib import Path
M, bad = Path(sys.argv[1]), []
for rev in ("stage1-step2000-tokens9B", "stage1-step5000-tokens21B", "stage1-step20000-tokens84B",
            "stage1-step150000-tokens630B", "stage1-step928646-tokens3896B"):
    d = M / rev
    try:
        marker = (d / ".complete").read_text().strip()
        shards = sorted(set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values()))
        missing = [f for f in shards if not (d / f).is_file() or (d / f).stat().st_size == 0]
        ok = marker == rev and bool(shards) and not missing
        print(f"  {'ok  ' if ok else 'BAD '} {rev}: .complete {marker!r}, {len(shards)} indexed shards, missing or empty {missing}")
    except (OSError, KeyError, ValueError) as e:
        ok = False
        print(f"  BAD  {rev}: {type(e).__name__}: {e}")
    if not ok:
        bad.append(rev)
sys.exit(3 if bad else 0)
PY
cat "$R"
[ "$bad" = 0 ] || { echo "## a digest differs or a checkpoint is missing"; exit 3; }
