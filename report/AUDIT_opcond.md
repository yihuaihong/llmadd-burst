# AUDIT: operation-conditioned priors (tools/opcond_train.py), 2026-10-13

Fresh adversarial reviewer (read-only; CPU checks incl. --tiny). Fixed before any job started (snapshot refreshed):
- BLOCKING (fixed): the first operand a sits at position 3, before the operator (position 4); with causal attention its
  state is identical in add / sub / mul prompts and equals the shared "Q: a" state (checked: max difference 0). Half of
  every "in-context" prior was therefore the shared prior. Now only operand b (position 6, after the operator) is shaped.
- Mul test is mostly recall: 292 of 500 mul test pairs have their swap trained. F52's "mul -12" splits into -4.5 on
  strict pairs and -17.0 on swap-trained pairs (42B). Main metrics: mul/test_strict and mul/train.
- prefix_helix does not reproduce F52's training (per-operation batches, summed CEs, 280 steps, mul ~5 epochs); only
  paired comparisons within this script are claimed. First check: does prefix_helix - none reproduce "+add, +sub,
  -mul_strict" here?
- Mixing a and b states in one Gram pressed out slot identity and the interaction in b (gone with b-only).
- Lambda adds up with the number of priors; matched shuffled controls were missing -> @norm arm, a shuffled arm for each
  prior set, step-0 gradient norms / cosines, 5 seeds for the key arms.
- Per-batch standardisation weakened the targets (factor: true vs shuffled CKA 0.45-0.64 within a batch) -> targets
  standardised over all values 10..99, a separate draw of 64 prompts per prior per step.
- No GPT-3-format eval (F53's only positive factor result was there): not comparable on that axis.
