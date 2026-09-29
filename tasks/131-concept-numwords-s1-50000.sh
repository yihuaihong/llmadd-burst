# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 080b-dl-early2
# Concept domain (ext 5): number words (one..twenty, thirty..ninety): same geometry as digits, rarer surface form; stage1-step50000-tokens210B; LoRA, lam=20, 3 seeds.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --domain numwords --lam 20 --seeds 0,1,2 --save_preds
