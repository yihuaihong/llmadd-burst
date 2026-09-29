# kind: gpu
# time: 02:00:00
# cpus: 8
# after: 210-dl-pythia-1.4b
# Second model family: Pythia-1.4b at step 64000 (128B tokens); manifold_train protocol (LoRA, lam=20, 3 seeds,
# CKA at 1/8..1/2 of the depth), prompt "Q: a + b =" -> " a+b" (GPT-NeoX tokenizer).
set -eu
cd tools && "$PY" pythia_train.py --model "$MODELS/pythia-1.4b/step64000" --out "$OUT" --lam 20 --geoms none,helix,helix_shuf,digit --seeds 0,1,2
