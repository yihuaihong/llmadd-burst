# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 004-exp1-main
# Idea 1 follow-up on main: is the tens digit coded separately from the helix? (digit bases, block edits)
set -eu
cd tools && "$PY" exp1b_digits.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
