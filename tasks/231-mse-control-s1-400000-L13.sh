# kind: gpu
# time: 04:00:00
# cpus: 8
# after: 080b-dl-early2
# Controls for the collaborator's point-wise pull towards main (Tab 1, 09/23): LoRA on A+-B, MSE at layer 13 to main's
# operand states (operand identity / running result) vs shuffled main, self-anchor and a random-number target; eval on
# unseen two-term and on three-term A+B+C / A-B-C. stage1-step400000-tokens1678B, lam=1, 3 seeds.
set -eu
cd tools && "$PY" mse_control.py --model "$MODELS/OLMo-2-1124-7B/stage1-step400000-tokens1678B" --main_model "$MODELS/OLMo-2-1124-7B/main" --out "$OUT" --layer 13 --lam 1 --seeds 0,1,2
