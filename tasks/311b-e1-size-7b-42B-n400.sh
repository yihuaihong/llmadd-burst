# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E1: is the window set by representation maturity or by task headroom? 7B 42B, 400 train pairs, 11 epochs.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step10000-tokens42B" --out "$OUT" --mode lora --lam 20 --fixed_test --train_pairs 400 --epochs 11 --save_preds --geoms none,helix --seeds 0,1,2
