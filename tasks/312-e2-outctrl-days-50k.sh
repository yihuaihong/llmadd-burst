# kind: gpu
# time: 03:00:00
# cpus: 8
# after: 006b-dl-early 007b-dl-mid 080b-dl-early2
# E2: answer-circle controls (permuted circle; one-hot clusters), plam 5, resampled batch, 5 seeds.
set -eu
cd tools && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT" --domain days --lam 20 --plam 5 --presample --seeds 0,1,2,3,4 --save_preds --geoms out_circle_shuf,out_onehot && "$PY" concept_train.py --model "$MODELS/OLMo-2-1124-7B/stage1-step50000-tokens210B" --out "$OUT/oc34" --domain days --lam 20 --plam 5 --presample --seeds 3,4 --save_preds --geoms out_circle
