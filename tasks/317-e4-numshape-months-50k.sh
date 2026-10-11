# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E4: number-manifold shaping transferred to the concept task, and number-subspace ablation at the item.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --domain months --lam 20 --seeds 0,1,2 --save_preds --geoms none,num_helix,num_helix_shuf --ablate none
