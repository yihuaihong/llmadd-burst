# kind: gpu
# gpus: 2
# time: 02:00:00
# cpus: 16
# after: 062-env-bnb
# Smoke test: FSDP full FT, 8-bit AdamW, 2 x A100 40GB, one geometry, one seed, one epoch (memory + speed).
set -eu
nvidia-smi --query-gpu=name,memory.total --format=csv
cd tools && "$PY" -m torch.distributed.run --nproc_per_node=2 manifold_train_fsdp.py --model "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" \
  --out "$OUT" --optim adamw8bit --geoms helix --seeds 0 --epochs 1
