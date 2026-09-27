# kind: gpu
# time: 01:00:00
# cpus: 8
# after: 020-probes-main
# Idea 2 follow-up on main: does an a+b readout trained at '=' of "a+b=" transfer to the second '+' of "a+b+c="?
set -eu
cd tools && "$PY" exp2c_transfer.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
