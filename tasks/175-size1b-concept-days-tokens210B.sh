# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 157-dl-1b
# Model-size study (step 2): concept domain days on OLMo-2-0425-1B stage1-step100000-tokens210B; same protocol as tasks 139-150 (LoRA, lam=20, 3 seeds),
# geometry at layers 2,4,6,8 (1/8..1/2 of the depth).
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-0425-1B/stage1-step100000-tokens210B" --out "$OUT" --domain days --lam 20 --layers 2,4,6,8 --seeds 0,1,2 --save_preds
