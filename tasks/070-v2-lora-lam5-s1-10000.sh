# kind: gpu
# time: 05:00:00
# cpus: 8
# after: 064-v2-lora-s1-10000
# Stronger geometry constraint (lam=5) with LoRA on s1-10000: does a manifold actually imposed change generalisation?
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" --out "$OUT" --mode lora --lam 5 --geoms helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
