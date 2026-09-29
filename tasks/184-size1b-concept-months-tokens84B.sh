# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 157-dl-1b
# Model-size study (step 2): concept domain months on OLMo-2-0425-1B stage1-step40000-tokens84B; same protocol as tasks 139-150 (LoRA, lam=20, 3 seeds),
# geometry at layers 2,4,6,8 (1/8..1/2 of the depth).
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-0425-1B/stage1-step40000-tokens84B" --out "$OUT" --domain months --lam 20 --layers 2,4,6,8 --seeds 0,1,2 --save_preds
