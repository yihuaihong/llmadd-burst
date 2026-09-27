# kind: gpu
# time: 00:50:00
# cpus: 8
# after: 043-readout-main
# Robustness of the tens readout error across prompt formats; two-term tens profile; unembedding geometry. main
set -eu
cd tools && "$PY" exp3_robust.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
