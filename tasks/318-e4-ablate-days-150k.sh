# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E4: number-subspace ablation at the item position on trained 150k models.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --domain days --lam 20 --seeds 0,1,2 --geoms none --ablate none
