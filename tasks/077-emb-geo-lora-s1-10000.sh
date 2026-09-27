# kind: gpu
# time: 05:00:00
# cpus: 8
# after: 071-v2-lora-lam20-s1-10000
# Embedding-site geometry (lam=20): LoRA + trainable number-token embedding rows, CKA loss on the embedding rows themselves. s1-10000.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" --out "$OUT" --mode lora --emb_delta --geo_site emb --lam 20 --geoms none,helix,helix_shuf,digit,digit_shuf --seeds 0,1,2
