# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 080b-dl-early2
# Concept domain (ext 5): letters A..Z (linear, 26): line vs shuffled line vs circle (wrong topology); stage1-step150000-tokens630B; LoRA, lam=20, 3 seeds.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --domain letters --lam 20 --seeds 0,1,2 --save_preds
