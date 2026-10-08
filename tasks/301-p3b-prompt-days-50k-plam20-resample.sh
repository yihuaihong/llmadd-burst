# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# P3b (F37 follow-up): same weight as 283/286 (20) but a fresh batch every step - isolates the resampling.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --domain days --lam 20 --plam 20 --presample --seeds 0,1,2 --save_preds --geoms none,sum_helix,sum_helix_shuf,out_circle
