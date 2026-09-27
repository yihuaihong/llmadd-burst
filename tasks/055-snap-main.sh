# kind: gpu
# time: 01:00:00
# cpus: 8
# after: 046-robust-main
# Manifold snapping at inference: do on-manifold operand representations improve accuracy? main
set -eu
cd tools && "$PY" exp4_snap.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
