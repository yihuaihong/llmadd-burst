# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E3: does raising kappa (sep_ce, first half) unlock the helix gain early? 7B 9B.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step2000-tokens9B" --out "$OUT" --mode lora --lam 20 --sep_lam 5 --save_preds --geoms none,helix,sep_ce,sep_then_helix --seeds 0,1,2
