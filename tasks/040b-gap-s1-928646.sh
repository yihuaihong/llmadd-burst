# kind: gpu
# time: 00:40:00
# cpus: 8
# after: 024-exp2c-main
# after_ended: 040-gap-s1-928646
# Per-problem gap: probe-decoded total at '=' vs the model's answer, digit by digit, stage1-step928646-tokens3896B.
set -eu
cd tools && "$PY" exp2e_gap.py --model "$MODELS/OLMo-2-1124-7B/stage1-step928646-tokens3896B" --out "$OUT"
