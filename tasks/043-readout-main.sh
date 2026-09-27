# kind: gpu
# time: 00:40:00
# cpus: 8
# after: 042-gap-main
# Where the t-10 answers come from: digit error structure, logit lens, tens repair edit, main.
set -eu
cd tools && "$PY" exp2f_readout.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
