# kind: gpu
# time: 01:00:00
# cpus: 8
# after_ended: 009-sweep-s1-150
# Re-run the exp2 probes with folds grouped by (a,b) pair (the first run could memorise pairs).
set -eu
cd tools && "$PY" exp2_three.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150-tokens1B" --out "$OUT" --probes_only
