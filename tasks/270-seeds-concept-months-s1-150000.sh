# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 080b-dl-early2
# Extra seeds (3, 4) for the cyclic domains on 7B (tasks 139-150 were noisy); pooled for statistics.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --domain months --lam 20 --seeds 3,4 --save_preds
