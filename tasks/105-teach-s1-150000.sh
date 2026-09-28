# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Learned target (ext 1): pull the operand geometry of stage1-step150000-tokens630B towards MAIN's own geometry of 10..99 (per layer), vs shuffled main and a self-anchor; lam=20, LoRA.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step150000-tokens630B" --out "$OUT" --mode lora --lam 20 --save_preds --main_model "$MODELS/OLMo-2-1124-7B/main" --geoms main,main_shuf,self --seeds 0,1,2
