# kind: gpu
# time: 06:00:00
# cpus: 8
# after: 080-dl-early2
# Maturity curve: task-only + strong geometry (lam=20) with LoRA on stage1-step20000-tokens84B.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step20000-tokens84B" --out "$OUT" --mode lora --lam 20 --geoms none,helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
