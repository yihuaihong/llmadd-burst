# kind: cpu
# time: 16:00:00
# cpus: 8
# after: 001-env
# after_ended: 008-dl-stage2
# Download + bf16-convert OLMo-2 checkpoints for the developmental sweep (CPU: no idle GPU).
set -eu
for rev in stage2-ingredient1-step11931-tokens50B stage2-ingredient2-step11931-tokens50B stage2-ingredient3-step11931-tokens50B; do
  "$PY" tools/fetch_olmo.py --revision "$rev" --dest "$MODELS/OLMo-2-1124-7B/$rev"
done
du -sh "$MODELS"/OLMo-2-1124-7B/* | tee "$OUT/models.txt"
df -h "/scratch/$USER" | tail -1 | tee -a "$OUT/models.txt"
