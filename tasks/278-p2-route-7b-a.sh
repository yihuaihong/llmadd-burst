# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# P2: number route (after arXiv 2605.01148): does "Friday plus 3 days" go through the base-10 number line?
# Sum probe fitted on "Q: {a} plus {k} {unit} is", applied to days / months / letters / number words; operand
# route; item-number alignment of embeddings and states, all with permutation nulls.
set -eu
cd tools && "$PY" number_route.py --out "$OUT" "$MODELS/OLMo-2-1124-7B/stage1-step2000-tokens9B" "$MODELS/OLMo-2-1124-7B/stage1-step5000-tokens21B" "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B"
