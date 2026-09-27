# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 008-dl-stage2 004-exp1-main 005-exp2-main
# Developmental sweep: exp1 + exp2 on stage2-ingredient3-step11931-tokens50B (same code and settings as on main).
set -eu
M="$MODELS/OLMo-2-1124-7B/stage2-ingredient3-step11931-tokens50B"
cd tools
"$PY" exp1_steer.py --model "$M" --out "$OUT/exp1"
"$PY" exp2_three.py --model "$M" --out "$OUT/exp2"
