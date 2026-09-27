# kind: cpu
# time: 06:00:00
# cpus: 8
# after: 001-env
# Early checkpoints for the "manifold maturity vs constraint gain" curve.
set -eu
for rev in stage1-step2000-tokens9B stage1-step5000-tokens21B stage1-step20000-tokens84B; do
  "$PY" tools/fetch_olmo.py --revision "$rev" --dest "$MODELS/OLMo-2-1124-7B/$rev"
done
du -sh "$MODELS"/OLMo-2-1124-7B/* | tee "$OUT/models.txt"
