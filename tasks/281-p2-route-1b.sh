# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 157-dl-1b
# P2: number route (after arXiv 2605.01148): does "Friday plus 3 days" go through the base-10 number line?
# Sum probe fitted on "Q: {a} plus {k} {unit} is", applied to days / months / letters / number words; operand
# route; item-number alignment of embeddings and states, all with permutation nulls.
set -eu
cd tools && "$PY" number_route.py --out "$OUT" "$MODELS/OLMo-2-0425-1B/stage1-step10000-tokens21B" "$MODELS/OLMo-2-0425-1B/stage1-step20000-tokens42B" "$MODELS/OLMo-2-0425-1B/stage1-step40000-tokens84B" "$MODELS/OLMo-2-0425-1B/stage1-step100000-tokens210B" "$MODELS/OLMo-2-0425-1B/stage1-step300000-tokens630B" "$MODELS/OLMo-2-0425-1B/stage1-step800000-tokens1678B"
