# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 071-v2-lora-lam20-s1-10000
# Embedding transplant (main_shuf): number-token rows of main mapped into s1-10000 by Procrustes on non-number tokens, then task-only LoRA. Compare with 064 (orig).
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" --out "$OUT" --mode lora --emb_init main_shuf --main_model "$MODELS/OLMo-2-1124-7B/main" --geoms none --seeds 0,1,2
