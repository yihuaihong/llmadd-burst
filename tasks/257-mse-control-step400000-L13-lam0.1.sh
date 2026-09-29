# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Pull-towards-main controls, weaker pull (lam=0.1); lam=1 destroyed three-term generalisation for every target (F31).
set -eu
cd tools && "$PY" mse_control.py --model "$MODELS/OLMo-2-1124-7B/stage1-step400000-tokens1678B" --main_model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT" --layer 13 --lam 0.1 --seeds 0,1,2
