# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Other task (ext 4): subtraction a - b (a >= b), random split; stage1-step5000-tokens21B, lam=20, LoRA.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step5000-tokens21B" --out "$OUT" --mode lora --lam 20 --save_preds --task sub --geoms none,helix,helix_shuf,digit --seeds 0,1,2
