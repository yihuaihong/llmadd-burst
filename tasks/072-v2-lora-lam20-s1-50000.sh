# kind: gpu
# time: 05:00:00
# cpus: 8
# after: 071-v2-lora-lam20-s1-10000
# Strong geometry constraint (lam=20, lora) on stage1-step50000-tokens210B: does the s1-10000 gain replicate / grow?
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --mode lora --lam 20 --geoms helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
