# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 157-dl-1b
# Model-size study (step 3): inference-time snap / shuffle of operand coordinates (exp4, F8) on OLMo-2-0425-1B stage1-step100000-tokens210B,
# layers 0-10 (the first ~2/3 of the depth, as 0-20 of 32 for 7B).
set -eu
cd tools && "$PY" exp4_snap.py --model "$MODELS/OLMo-2-0425-1B/stage1-step100000-tokens210B" --out "$OUT" --layers 0-10 --step 1
