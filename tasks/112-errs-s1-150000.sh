# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Error anatomy (ext 2): why do the 10..999 geometries hurt three-digit extrapolation on stage1-step150000? lam=20, LoRA, predictions saved.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --mode lora --lam 20 --save_preds --geoms none,helix,helix3,helix3_shuf,digit3 --seeds 0,1,2
