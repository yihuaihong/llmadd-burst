# kind: gpu
# time: 06:00:00
# cpus: 8
# after: 159-dl-13b-b
# Model-size study (step 2): concept domain months on OLMo-2-1124-13B stage1-step5000-tokens42B; same protocol as tasks 139-150 (LoRA, lam=20, 3 seeds),
# geometry at layers 5,10,15,20 (1/8..1/2 of the depth).
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-13B/stage1-step5000-tokens42B" --out "$OUT" --domain months --lam 20 --layers 5,10,15,20 --seeds 0,1,2 --save_preds
