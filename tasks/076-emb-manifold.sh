# kind: cpu
# time: 06:00:00
# cpus: 8
# after: 001-env
# Number manifold in the embedding / unembedding matrices over 19 checkpoints (only 2 shards per revision downloaded).
set -eu
cd tools && "$PY" emb_manifold.py --out "$OUT"
