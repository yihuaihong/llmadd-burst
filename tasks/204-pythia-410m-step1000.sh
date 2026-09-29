# kind: gpu
# time: 01:30:00
# cpus: 8
# after: 203-dl-pythia-410m
# Second model family: Pythia-410m at step 1000 (2B tokens); manifold_train protocol (LoRA, lam=20, 3 seeds,
# CKA at 1/8..1/2 of the depth), prompt "Q: a + b =" -> " a+b" (GPT-NeoX tokenizer).
set -eu
cd tools && "$PY" pythia_train.py --model "$MODELS/pythia-410m/step1000" --out "$OUT" --lam 20 --geoms none,helix,helix_shuf,digit --seeds 0,1,2
