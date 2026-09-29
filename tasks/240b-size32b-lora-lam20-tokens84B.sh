# kind: gpu
# gpus: 2
# time: 12:00:00
# cpus: 16
# after: 236-dl-32b-b
# after_ended: 240-size32b-lora-lam20-tokens84B
# Model-size study: OLMo-2-0325-32B stage1-step10000-tokens84B, split over 2 x A100-40GB (--device_map auto); same protocol as 1B/7B/13B
# (LoRA, lam=20, none/helix/helix_shuf/digit, 3 seeds), geometry at layers 8,16,24,32 (1/8..1/2 of 64).
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-0325-32B/stage1-step10000-tokens84B" --out "$OUT" --mode lora --lam 20 --layers 8,16,24,32 --device_map auto --save_preds --geoms none,helix,helix_shuf,digit --seeds 0,1,2
