# kind: gpu
# time: 06:00:00
# cpus: 8
# after: 157-dl-1b
# Extra seeds (3, 4) for the size study's benefit window (OLMo-2-0425-1B stage1-step40000-tokens84B); pooled with tasks 161-172 for statistics.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-0425-1B/stage1-step40000-tokens84B" --out "$OUT" --mode lora --lam 20 --layers 2,4,6,8 --save_preds --geoms none,helix,helix_shuf,digit --seeds 3,4
