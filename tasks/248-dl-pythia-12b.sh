# kind: cpu
# time: 06:00:00
# cpus: 8
# after: 001-env
# Second model family: download EleutherAI/pythia-12b at steps 1000 4000 16000 32000 64000 143000 (LFS hashes checked distinct on 2026-09-30).
set -eu
for st in step1000 step4000 step16000 step32000 step64000 step143000; do
  "$PY" tools/fetch_hf.py --repo EleutherAI/pythia-12b --revision "$st" --dest "$MODELS/pythia-12b/$st"
done
du -sh "$MODELS"/pythia-12b/* | tee "$OUT/models.txt"
