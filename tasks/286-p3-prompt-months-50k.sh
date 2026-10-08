# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# P3: prompt-state targets for the cyclic domains (same protocol as 139-150, LoRA, lam 20, 280 steps, 3 seeds):
# sum_helix (final token, 3/8..5/8 depth -> helix of the pre-mod sum i +- k), sum_helix_shuf, out_circle (3/4, 7/8
# depth -> circle of the answer), against none and the item circle. Eval: test + ood (k 41..80).
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --domain months --lam 20 --seeds 0,1,2 --save_preds --geoms none,circle,sum_helix,sum_helix_shuf,out_circle
