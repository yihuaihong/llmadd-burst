# kind: cpu
# time: 12:00:00
# cpus: 8
# after: 080b-dl-early2
# Model-size study: OLMo-2-0325-32B at 210B / 630B tokens (the 84B point was still on the rising side, F32).
set -eu
for rev in stage1-step25000-tokens210B stage1-step75000-tokens630B; do
  "$PY" tools/fetch_olmo.py --repo allenai/OLMo-2-0325-32B --revision "$rev" --dest "$MODELS/OLMo-2-0325-32B/$rev"
done
du -sh "$MODELS"/OLMo-2-0325-32B/* | tee "$OUT/models.txt"
