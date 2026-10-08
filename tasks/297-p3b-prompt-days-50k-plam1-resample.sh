# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# P3b (F37 follow-up): prompt-state targets with a smaller weight and a fresh 96-prompt batch every step.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --domain days --lam 20 --plam 1 --presample --seeds 0,1,2 --save_preds --geoms none,sum_helix,sum_helix_shuf,out_circle
