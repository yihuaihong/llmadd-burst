# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 003-gpu-smoke
# Idea 2 on main: probes / interchange / steering of the running sum in A+B+C.
set -eu
cd tools && "$PY" exp2_three.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
