# kind: gpu
# time: 08:00:00
# cpus: 8
# after: 159-dl-13b-b
# Model-size study (step 1): OLMo-2-1124-13B stage1-step5000-tokens42B; same protocol as the 7B maturity curve (F14): LoRA, lam=20,
# geometry at layers 5,10,15,20 (1/8, 1/4, 3/8, 1/2 of the depth), none/helix/helix_shuf/digit, 3 seeds.
set -eu
cd tools && "$PY" manifold_train.py --model "$MODELS/OLMo-2-1124-13B/stage1-step5000-tokens42B" --out "$OUT" --mode lora --lam 20 --layers 5,10,15,20 --save_preds --geoms none,helix,helix_shuf,digit --seeds 0,1,2
