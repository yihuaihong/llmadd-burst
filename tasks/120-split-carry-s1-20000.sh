# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Systematic generalisation (ext 3): train without a units carry, test only on carry pairs; stage1-step20000-tokens84B, lam=20, LoRA.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step20000-tokens84B" --out "$OUT" --mode lora --lam 20 --save_preds --split carry --geoms none,helix,helix_shuf,digit --seeds 0,1,2
