# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 080b-dl-early2
# Concept domain (ext 5): months (cyclic, 12): circle vs shuffled circle vs line (wrong topology); stage1-step20000-tokens84B; LoRA, lam=20, 3 seeds.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step20000-tokens84B" --out "$OUT" --domain months --lam 20 --seeds 0,1,2 --save_preds
