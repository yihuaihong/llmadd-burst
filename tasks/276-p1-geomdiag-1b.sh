# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 157-dl-1b
# P1 (related_papers_2026-10-07.md): spectral (Fourier share at 1/T) vs geometric (CV kappa of n mod T) structure
# of the number representations (embedding rows, "Q: x" states at 1/8..3/4 depth), and digits vs number words 1..20.
set -eu
cd tools && "$PY" geom_diag.py --out "$OUT" "$MODELS/OLMo-2-0425-1B/stage1-step10000-tokens21B" "$MODELS/OLMo-2-0425-1B/stage1-step20000-tokens42B" "$MODELS/OLMo-2-0425-1B/stage1-step40000-tokens84B" "$MODELS/OLMo-2-0425-1B/stage1-step100000-tokens210B" "$MODELS/OLMo-2-0425-1B/stage1-step300000-tokens630B" "$MODELS/OLMo-2-0425-1B/stage1-step800000-tokens1678B"
