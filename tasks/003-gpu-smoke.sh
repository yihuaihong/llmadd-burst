# kind: gpu
# time: 00:40:00
# cpus: 8
# after: 002-models-main
# First GPU task: env + model load work on the A100, and the position-0 vs in-context norm check.
set -eu
"$PY" tools/gpu_smoke.py --model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT"
