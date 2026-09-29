# kind: cpu
# time: 06:00:00
# cpus: 8
# after: 001-env
# Second model family: download EleutherAI/pythia-1.4b at steps 1000 4000 16000 32000 64000 143000 (2M tokens per step).
set -eu
for st in step1000 step4000 step16000 step32000 step64000 step143000; do
  "$PY" tools/fetch_hf.py --repo EleutherAI/pythia-1.4b --revision "$st" --dest "$MODELS/pythia-1.4b/$st"
done
du -sh "$MODELS"/pythia-1.4b/* | tee "$OUT/models.txt"
