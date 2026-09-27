# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 005-exp2-main
# Idea 2 follow-up on main: which positions the answer reads a, b from (interchange by outcome, knockout).
set -eu
cd tools && "$PY" exp2b_route.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
