# kind: cpu
# time: 12:00:00
# cpus: 8
# after: 080b-dl-early2
# Model-size study: download allenai/OLMo-2-0325-32B revisions (129 GB fp32 each on the Hub -> 65 GB bf16 copy only).
set -eu
for rev in stage1-step1000-tokens9B stage1-step2000-tokens17B; do
  "$PY" tools/fetch_olmo.py --repo allenai/OLMo-2-0325-32B --revision "$rev" --dest "$MODELS/OLMo-2-0325-32B/$rev"
done
du -sh "$MODELS"/OLMo-2-0325-32B/* | tee "$OUT/models.txt"
df -h "/scratch/$USER" | tail -1 | tee -a "$OUT/models.txt"
