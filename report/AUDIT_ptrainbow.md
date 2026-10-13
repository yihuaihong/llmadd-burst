# AUDIT: rainbow step 2 in pretraining (tools/pretrain_rainbow.py), 2026-10-13

Fresh adversarial reviewer (read-only; 76,654 shaped positions checked, gradients measured on the real 53M model on CPU).
Positions, add split, scoring index, single-token answers, @norm, sbatch quoting: fine. Fixed before any job started:
- BLOCKING: days / months "held-out" combos were aliased: 100% of held-out days problems had a trained combo with the
  same item and effective shift (Fri + 9 vs Fri + 2). -> held out by (item, effective shift) class (e != 0): 13 of 42
  day classes, 40 of 132 month classes; a held-out problem needs the shift learned on other items.
- Prior strengths differ: lam * |grad| at init helix 6.6, days 18.0, months 13.1 (LM 2.9); @norm keeps the ratio.
  -> "@bal" arm: lam per prior rescaled at step 0 to the smallest prior's pull (smoke run: 5 / 1.7 / 2.65); gradient
  norms of the LM and all three priors and their cosines logged at every eval, for every arm.
- Shuffled targets only half wrong for some seeds (days seed 1: CKA 0.44) -> re-drawn until CKA < 0.15. Note: the
  none model's item states are already close to the true circles (CKA 0.63 / 0.67): "shuffled hurts" can mean "breaking
  a natural structure"; the claim that needs evidence is "true prior > none".
- The helix shapes the number tokens 10..20 that are also the shifts k -> days / months held-out split by k <= 9.
- Tied embeddings -> add_held split by sum < 100.
- No double measurement -> CKA of the shaped tokens in held-out prompts to all three true geometries, for every arm.
- Splits now drawn per seed; code sha256 written into every run file.
Open (documented, not fixed): eval prompts follow text while the last training line follows 5 solved lines; the
"in-context" states differ in kind (items / operand a see no task context, operand b sees "a +"); exposure counts are
high (each trained days combo ~6k times), so compare curves / onset, not only final accuracy.
