# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Systematic generalisation (ext 3): as split-holdout but the geometry loss sees only the 75 training operand values; stage1-step10000-tokens42B.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" --out "$OUT" --mode lora --lam 20 --save_preds --split holdout_operand --geo_numbers seen --geoms helix,helix_shuf --seeds 0,1,2
