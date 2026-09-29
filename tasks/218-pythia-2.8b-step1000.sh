# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 217-dl-pythia-2.8b
# Second model family: Pythia-2.8b at step 1000 (2B tokens); manifold_train protocol (LoRA, lam=20, 3 seeds,
# CKA at 1/8..1/2 of the depth), prompt "Q: a + b =" -> " a+b" (GPT-NeoX tokenizer).
set -eu
cd tools && "$PY" pythia_train.py --model "$MODELS/pythia-2.8b/step1000" --out "$OUT" --lam 20 --geoms none,helix,helix_shuf,digit --seeds 0,1,2
