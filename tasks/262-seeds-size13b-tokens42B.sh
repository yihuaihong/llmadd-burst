# kind: gpu
# time: 06:00:00
# cpus: 8
# after: 159-dl-13b-b
# Extra seeds (3, 4) for the size study's benefit window (OLMo-2-1124-13B stage1-step5000-tokens42B); pooled with tasks 161-172 for statistics.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-13B/stage1-step5000-tokens42B" --out "$OUT" --mode lora --lam 20 --layers 5,10,15,20 --save_preds --geoms none,helix,helix_shuf,digit --seeds 3,4
