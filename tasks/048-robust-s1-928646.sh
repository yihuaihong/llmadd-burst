# kind: gpu
# time: 00:50:00
# cpus: 8
# after: 046-robust-main
# exp3 on stage1-step928646-tokens3896B: is the stage-1 three-term failure also a format effect?
set -eu
cd tools && "$PY" exp3_robust.py --model "$MODELS/OLMo-2-1124-7B/stage1-step928646-tokens3896B" --out "$OUT"
