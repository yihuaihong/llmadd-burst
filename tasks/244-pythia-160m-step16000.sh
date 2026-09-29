# kind: gpu
# time: 01:00:00
# cpus: 8
# after: 241-dl-pythia-160m
# Second model family: Pythia-160m at step 16000 (32B tokens); same protocol as tasks 204-230.
set -eu
cd tools && "$PY" pythia_train.py --model "$MODELS/pythia-160m/step16000" --out "$OUT" --lam 20 --geoms none,helix,helix_shuf,digit --seeds 0,1,2
