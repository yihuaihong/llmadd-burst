# kind: cpu
# time: 12:00:00
# cpus: 8
# after: 080b-dl-early2
# Model-size study: download allenai/OLMo-2-0425-1B revisions (fp32 on the Hub -> bf16 copy only).
set -eu
for rev in stage1-step10000-tokens21B stage1-step20000-tokens42B stage1-step40000-tokens84B stage1-step100000-tokens210B stage1-step300000-tokens630B stage1-step800000-tokens1678B; do
  "$PY" tools/fetch_olmo.py --repo allenai/OLMo-2-0425-1B --revision "$rev" --dest "$MODELS/OLMo-2-0425-1B/$rev"
done
du -sh "$MODELS"/OLMo-2-0425-1B/* | tee "$OUT/models.txt"
df -h "/scratch/$USER" | tail -1 | tee -a "$OUT/models.txt"
