# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E5: item <-> number alignment for concepts with (planets, elements, presidents, Greek) or without (solfege) a textual numbering.
set -eu
cd tools && "$PY" number_route.py --out "$OUT" --domains planets,elements,presidents,greek,ordinals,solfege "$MODELS/OLMo-2-1124-7B/stage1-step2000-tokens9B" "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" "$MODELS/OLMo-2-1124-7B/stage1-step400000-tokens1678B" "$MODELS/OLMo-2-1124-7B/main"
