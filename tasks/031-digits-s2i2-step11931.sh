# kind: gpu
# time: 01:30:00
# cpus: 8
# after_ended: 016-sweep-s2i2-step11931
# exp1b (digit-wise code vs helix) on stage2-ingredient2-step11931-tokens50B, for the developmental curve.
set -eu
cd tools && "$PY" exp1b_digits.py --model "$MODELS/OLMo-2-1124-7B/stage2-ingredient2-step11931-tokens50B" --out "$OUT"
