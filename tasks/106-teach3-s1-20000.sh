# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Learned target over 10..999 (ext 1) on stage1-step20000-tokens84B: main3 / main3_shuf / self3; lam=20, LoRA.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step20000-tokens84B" --out "$OUT" --mode lora --lam 20 --save_preds --main_model "$MODELS/OLMo-2-1124-7B/main" --geoms main3,main3_shuf,self3 --seeds 0,1,2
