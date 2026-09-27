# kind: gpu
# time: 00:30:00
# cpus: 8
# after: 024-exp2c-main
# What the model says on a+b+c= (format vs knowledge), main.
set -eu
cd tools && "$PY" exp2d_outputs.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
