# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 158-dl-13b-a 159-dl-13b-b 160-dl-13b-c
# P1 (related_papers_2026-10-07.md): spectral (Fourier share at 1/T) vs geometric (CV kappa of n mod T) structure
# of the number representations (embedding rows, "Q: x" states at 1/8..3/4 depth), and digits vs number words 1..20.
set -eu
cd tools && "$PY" geom_diag.py --out "$OUT" "$MODELS/OLMo-2-1124-13B/stage1-step1000-tokens9B" "$MODELS/OLMo-2-1124-13B/stage1-step2000-tokens17B" "$MODELS/OLMo-2-1124-13B/stage1-step5000-tokens42B" "$MODELS/OLMo-2-1124-13B/stage1-step10000-tokens84B" "$MODELS/OLMo-2-1124-13B/stage1-step25000-tokens210B" "$MODELS/OLMo-2-1124-13B/stage1-step75000-tokens630B"
