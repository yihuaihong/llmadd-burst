# kind: gpu
# time: 01:30:00
# cpus: 8
# after_ended: 011-sweep-s1-50000
# exp1b (digit-wise code vs helix) on stage1-step50000-tokens210B, for the developmental curve.
set -eu
cd tools && "$PY" exp1b_digits.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT"
