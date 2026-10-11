# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E1: is the window set by representation maturity or by task headroom? 7B 9B, 7000 train pairs, 3 epochs.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step2000-tokens9B" --out "$OUT" --mode lora --lam 20 --fixed_test --train_pairs 7000 --epochs 3 --save_preds --geoms none,helix --seeds 0,1,2
