# kind: gpu
# time: 06:00:00
# cpus: 8
# after: 248-dl-pythia-12b
# Second model family: Pythia-12b at step 4000 (8B tokens); same protocol as tasks 204-230.
set -eu
cd tools && "$PY" pythia_train.py --model "$MODELS/pythia-12b/step4000" --out "$OUT" --lam 20 --geoms none,helix,helix_shuf,digit --seeds 0,1,2
