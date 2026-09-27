# kind: gpu
# time: 00:30:00
# cpus: 8
# after: 024-exp2c-main
# What the model says on a+b+c= (format vs knowledge), stage1-step400000-tokens1678B.
set -eu
cd tools && "$PY" exp2d_outputs.py --model "$MODELS/OLMo-2-1124-7B/stage1-step400000-tokens1678B" --out "$OUT"
