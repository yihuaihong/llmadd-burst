# kind: gpu
# time: 01:00:00
# cpus: 8
# after_ended: 005-exp2-main
# Re-run the exp2 probes with folds grouped by (a,b) pair (the first run could memorise pairs).
set -eu
cd tools && "$PY" exp2_three.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT" --probes_only
