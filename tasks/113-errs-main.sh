# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Error anatomy reference (ext 2): main itself, task-only LoRA, predictions saved.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT" --mode lora --lam 20 --save_preds --geoms none --seeds 0,1,2
