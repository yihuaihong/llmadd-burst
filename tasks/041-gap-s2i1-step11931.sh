# kind: gpu
# time: 00:40:00
# cpus: 8
# after: 024-exp2c-main
# Per-problem gap: probe-decoded total at '=' vs the model's answer, digit by digit, stage2-ingredient1-step11931-tokens50B.
set -eu
cd tools && "$PY" exp2e_gap.py --model "$MODELS/OLMo-2-1124-7B/stage2-ingredient1-step11931-tokens50B" --out "$OUT"
