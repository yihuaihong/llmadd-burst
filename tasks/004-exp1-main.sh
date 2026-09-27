# kind: gpu
# time: 02:30:00
# cpus: 8
# after: 003-gpu-smoke
# Idea 1 on main: in-context helix fits (held-out R^2 vs nulls) + phase-rotation steering at A and B.
set -eu
cd tools && "$PY" exp1_steer.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
