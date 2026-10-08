# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# P2b: number route with a 4-shot prefix of solved prompts of the same domain (no numeric examples), so that the
# base checkpoints actually solve the task (zero-shot accuracy was <= 5%, F36). Same read-outs as 278-280.
set -eu
cd tools && "$PY" number_route.py --out "$OUT" --shots 4 "$MODELS/OLMo-2-1124-7B/stage1-step400000-tokens1678B" "$MODELS/OLMo-2-1124-7B/stage1-step928646-tokens3896B" "$MODELS/OLMo-2-1124-7B/main"
