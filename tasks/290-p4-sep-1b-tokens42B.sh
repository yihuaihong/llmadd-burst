# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 157-dl-1b
# P4: spectrum vs separability (Fu et al., arXiv 2604.20817) inside the benefit window: helix vs helix_nonsep (same
# Fourier spikes, n mod T not separable) vs helix_coarse (its clean part only) vs sep_ce (separability only, no shape)
# vs digit; LoRA, lam 20 (sep_ce: sep_lam 5), 3 seeds.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-0425-1B/stage1-step20000-tokens42B" --out "$OUT" --mode lora --lam 20 --sep_lam 5 --layers 2,4,6,8 --save_preds --geoms none,helix,helix_nonsep,helix_coarse,sep_ce,digit --seeds 0,1,2
