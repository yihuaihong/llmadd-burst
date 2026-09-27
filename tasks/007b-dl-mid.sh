# kind: cpu
# time: 16:00:00
# cpus: 8
# after: 001-env
# after_ended: 007-dl-mid
# Download + bf16-convert OLMo-2 checkpoints for the developmental sweep (CPU: no idle GPU).
set -eu
for rev in stage1-step150000-tokens630B stage1-step400000-tokens1678B stage1-step928646-tokens3896B; do
  "$PY" tools/fetch_olmo.py --revision "$rev" --dest "$MODELS/OLMo-2-1124-7B/$rev"
done
du -sh "$MODELS"/OLMo-2-1124-7B/* | tee "$OUT/models.txt"
df -h "/scratch/$USER" | tail -1 | tee -a "$OUT/models.txt"
