# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# P3b seeds 3,4 (F40 follow-up): sum_helix vs none at plam 5, resampled batch.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --domain months --lam 20 --plam 5 --presample --seeds 3,4 --save_preds --geoms none,sum_helix,sum_helix_shuf
