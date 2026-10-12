# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E1 control: is the +70 at 9B/7000 helix-specific? shuffled helix and digit geometry.
set -eu
cd tools \&\& "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step2000-tokens9B" --out "$OUT" --mode lora --lam 20 --fixed_test --train_pairs 7000 --save_preds --epochs 3 --geoms helix_shuf,digit --seeds 0,1,2
