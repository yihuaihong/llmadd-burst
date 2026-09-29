# kind: cpu
# time: 12:00:00
# cpus: 8
# after: 080b-dl-early2
# Model-size study: download allenai/OLMo-2-1124-13B revisions (fp32 on the Hub -> bf16 copy only).
set -eu
for rev in stage1-step25000-tokens210B stage1-step75000-tokens630B; do
  "$PY" tools/fetch_olmo.py --repo allenai/OLMo-2-1124-13B --revision "$rev" --dest "$MODELS/OLMo-2-1124-13B/$rev"
done
du -sh "$MODELS"/OLMo-2-1124-13B/* | tee "$OUT/models.txt"
df -h "/scratch/$USER" | tail -1 | tee -a "$OUT/models.txt"
