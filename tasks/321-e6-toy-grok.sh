# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E6: geometry prior switched on in a 2000-step window of grokking (Nanda et al. setting); 5 seeds as parallel processes.
set -eu
cd tools && for s in 0 1 2 3 4; do "$PY" toy_grok.py --out "$OUT" --seeds $s --arms none,always,w0,w1000,w2000,w4000,w6000,w8000,w12000,always_shuf,w2000_shuf > "$OUT/log_seed$s.txt" 2>&1 & done; wait; "$PY" -c "import toy_grok, pathlib; toy_grok.summarize(pathlib.Path('$OUT'))"
