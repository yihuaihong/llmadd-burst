# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# P2c: number route on LoRA-trained models that solve the task (concept_train --route): base + none / circle arms,
# seeds 0, 1; same protocol as 139-150 (lam 20, 280 steps).
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --domain months --lam 20 --seeds 0,1 --geoms none,circle --route none,circle
