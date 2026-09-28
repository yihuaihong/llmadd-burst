# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Soft transplant (ext 1): embedding rows of 10..99 in stage1-step10000-tokens42B pulled towards the geometry of MAIN's embedding rows (CKA, lam=20, LoRA + number-row delta); compare the hard transplant 090-095.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" --out "$OUT" --mode lora --lam 20 --save_preds --main_model "$MODELS/OLMo-2-1124-7B/main" --emb_delta --geo_site emb --geoms main,main_shuf --seeds 0,1,2
