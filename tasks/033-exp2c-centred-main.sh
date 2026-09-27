# kind: gpu
# time: 00:40:00
# cpus: 8
# after: 024-exp2c-main
# exp2c again with per-format re-centring (does the a+b direction transfer even if the offset does not?).
set -eu
cd tools && "$PY" exp2c_transfer.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
