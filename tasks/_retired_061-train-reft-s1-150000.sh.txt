# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 050-snap-s1-10000
# Manifold-regularised fine-tuning (reft): none / helix / helix_shuf / digit / digit_shuf x 3 seeds on stage1-step150000-tokens630B.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --mode reft --geoms none,helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
