# kind: cpu
# time: 12:00:00
# cpus: 8
# after: 001-env
# Download OLMo-2-1124-7B revisions from Hugging Face (public) and keep a bf16 copy only
# (fp32 ~29 GB -> bf16 ~15 GB per revision). Runs on CPU: a GPU node idle for 20 min is shut down.
set -eu
REVS="main"
for rev in $REVS; do
  "$PY" tools/fetch_olmo.py --revision "$rev" --dest "$MODELS/OLMo-2-1124-7B/$rev"
done
du -sh "$MODELS"/OLMo-2-1124-7B/* | tee "$OUT/models.txt"
df -h "/scratch/$USER" | tail -1 | tee -a "$OUT/models.txt"
