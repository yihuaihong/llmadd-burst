# kind: gpu
# time: 06:00:00
# cpus: 8
# after: 160-dl-13b-c
# Model-size study (step 2): concept domain days on OLMo-2-1124-13B stage1-step75000-tokens630B; same protocol as tasks 139-150 (LoRA, lam=20, 3 seeds),
# geometry at layers 5,10,15,20 (1/8..1/2 of the depth).
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-13B/stage1-step75000-tokens630B" --out "$OUT" --domain days --lam 20 --layers 5,10,15,20 --seeds 0,1,2 --save_preds
