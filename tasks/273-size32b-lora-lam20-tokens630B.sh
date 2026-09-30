# kind: gpu
# gpus: 2
# time: 12:00:00
# cpus: 16
# after: 271-dl-32b-c
# Model-size study: OLMo-2-0325-32B stage1-step75000-tokens630B on 2 x A100-40GB (--device_map auto, gradient checkpointing); same protocol as 237b-240b.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-0325-32B/stage1-step75000-tokens630B" --out "$OUT" --mode lora --lam 20 --layers 8,16,24,32 --device_map auto --save_preds --geoms none,helix,helix_shuf,digit --seeds 0,1,2
