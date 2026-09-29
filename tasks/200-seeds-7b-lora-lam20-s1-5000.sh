# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Extra seeds (3, 4) for the 7B benefit window (F14, tasks 081/082b/071/083); pooled with seeds 0-2 for the paper statistics.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step5000-tokens21B" --out "$OUT" --mode lora --lam 20 --save_preds --geoms none,helix,helix_shuf,digit --seeds 3,4
