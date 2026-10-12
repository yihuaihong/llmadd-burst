# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E1 control: does none catch up with 10 epochs instead of 3?
set -eu
cd tools \&\& "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step2000-tokens9B" --out "$OUT" --mode lora --lam 20 --fixed_test --train_pairs 7000 --save_preds --epochs 10 --geoms none --seeds 0,1
