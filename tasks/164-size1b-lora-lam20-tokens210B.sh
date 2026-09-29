# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 157-dl-1b
# Model-size study (step 1): OLMo-2-0425-1B stage1-step100000-tokens210B; same protocol as the 7B maturity curve (F14): LoRA, lam=20,
# geometry at layers 2,4,6,8 (1/8, 1/4, 3/8, 1/2 of the depth), none/helix/helix_shuf/digit, 3 seeds.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-0425-1B/stage1-step100000-tokens210B" --out "$OUT" --mode lora --lam 20 --layers 2,4,6,8 --save_preds --geoms none,helix,helix_shuf,digit --seeds 0,1,2
