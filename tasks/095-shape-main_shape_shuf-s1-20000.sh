# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 080b-dl-early2
# Shape-only transplant (main_shape_shuf) of main's number embeddings into stage1-step20000-tokens84B, then task-only LoRA.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step20000-tokens84B" --out "$OUT" --mode lora --emb_init main_shape_shuf --main_model "$MODELS/OLMo-2-1124-7B/main" --geoms none --seeds 0,1,2
