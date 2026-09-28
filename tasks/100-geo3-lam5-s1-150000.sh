# kind: gpu
# time: 06:00:00
# cpus: 8
# after: 080b-dl-early2
# 0..999 geometries (helix3 / digit3 + shuffled), lam=5, LoRA on stage1-step150000-tokens630B: does a three-digit target help extrapolation?
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --mode lora --lam 5 --geoms helix3,helix3_shuf,digit3,digit3_shuf --seeds 0,1,2
