# AUDIT: rainbow fine-tuning (tools/rainbow_train.py), 2026-10-13

Fresh adversarial reviewer (read-only, CPU checks incl. --tiny runs in two arm orders). No code bug found:
data and splits identical to manifold_train / concept_train (checked item by item), last-position states correct
("Q: 10" = 4 tokens, "Q: Monday" = 3), layer indices as in the single-domain runs, target Grams identical, LoRA reset
complete (two arm orders give identical logs), eval in eval mode with single-token answers, test_strict correct.

Interpretation risks and what was done (all before any rainbow job started; snapshot refreshed to a0f8641):
1. The helix prior is not orthogonal to the date tasks: the shift k in "Friday plus 45 days" is a number token, and
   F44 E4a showed the helix alone helps months (+11.4) and days (+12.5). -> days / months test split by k:
   test_k1to9 (single-digit k, not shaped by the helix prior) vs test_k10plus. "All vs single" and "all-but-one" must
   be read against the full 3 x 3 cross-effect matrix (each single prior on each task), not as additivity of
   orthogonal manifolds. Note test_k1to9 is small (days 48, months 81 items).
2. Lambda adds up with the number of active priors and the three tasks share one AdamW. -> arms "@norm" (lambda / number
   of priors); step-0 gradient norms and pairwise cosines of every CE and prior term on the LoRA parameters.
3. Not comparable with F14 / F24 numbers (multi-task, shared optimiser, step counts, autocast differed). -> autocast now
   off as in the single-domain LoRA scripts; only paired comparisons within this script are claimed.
4. Power: 3 seeds, 48 tests per checkpoint. -> key arms get seeds 3-4 (5 seeds); report per-seed differences.
5. Shuffled targets were only weakly wrong for some seeds (days seed 1: CKA 0.44 to the true circle). -> re-drawn
   until CKA < 0.15.
6. "ood" (k 41..80) is not out-of-distribution here (these numbers are add operands) -> renamed k41to80.
7. Days / months test items share item and k mod n with training items (design property, as in F24); add: use
   test_strict as the main metric (18.4% of test pairs have their swap trained).
