# kind: gpu
# time: 01:00:00
# cpus: 8
# after: 046-robust-main
# after_ended: 052-snap-s1-150000
# Manifold snapping at inference: do on-manifold operand representations improve accuracy? stage1-step150000-tokens630B
set -eu
cd tools && "$PY" exp4_snap.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT"
