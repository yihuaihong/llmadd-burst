# kind: cpu
# time: 16:00:00
# cpus: 8
# after: 001-env
# Download + bf16-convert OLMo-2 checkpoints for the developmental sweep (CPU: no idle GPU).
set -eu
for rev in stage1-step150-tokens1B stage1-step10000-tokens42B stage1-step50000-tokens210B; do
  "$PY" tools/fetch_olmo.py --revision "$rev" --dest "$MODELS/OLMo-2-1124-7B/$rev"
done
du -sh "$MODELS"/OLMo-2-1124-7B/* | tee "$OUT/models.txt"
df -h "/scratch/$USER" | tail -1 | tee -a "$OUT/models.txt"
