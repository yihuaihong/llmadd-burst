# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 159-dl-13b-b
# Model-size study (step 3): inference-time snap / shuffle of operand coordinates (exp4, F8) on OLMo-2-1124-13B stage1-step10000-tokens84B,
# layers 0-26 (the first ~2/3 of the depth, as 0-20 of 32 for 7B).
set -eu
cd tools && "$PY" exp4_snap.py --model "$MODELS/OLMo-2-1124-13B/stage1-step10000-tokens84B" --out "$OUT" --layers 0-26 --step 2
