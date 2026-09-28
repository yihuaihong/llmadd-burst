# kind: gpu
# time: 05:00:00
# cpus: 8
# after: 080b-dl-early2
# Embedding-site geometry curve (lam=20, LoRA + number-row delta) on stage1-step150000-tokens630B.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --mode lora --emb_delta --geo_site emb --lam 20 --geoms none,helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
