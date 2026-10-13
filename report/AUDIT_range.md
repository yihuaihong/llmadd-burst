# AUDIT: experiment B (tools/pretrain_range.py), 2026-10-13

Fresh adversarial reviewer (read-only; CPU simulations of the full training stream). No code bug: held-out pairs never
appear in 2.34M simulated training lines; scoring verified with fake models; operand positions verified on 4,800
spans; piecewise = whole-string tokenisation on 17,528 lines. Fixed before any job started (snapshot refreshed):
- BLOCKING for attribution: ~80% of the digitcirc target Gram is the number-of-digits grouping (CKA to a digit-count
  one-hot target 0.805), and digitcirc_shuf destroyed both. -> added digitcirc_lenshuf / helix3_lenshuf (values
  permuted within 0..9, 10..99, 100..999: keeps the grouping, destroys the geometry) and a length-only target.
- The old "train_mix" was not "trained" (only 18 of 52 three-digit pairs ever drawn). -> a rehearsal pool (100 pairs per
  bucket, 5% of the lines): seen_bXY are certainly trained.
- held_b11 had 28 items from one fixed split for all seeds -> 20 unordered pairs (~38 items), split drawn per seed;
  report Wilson intervals. Note: the digit facts of b11 occur inside larger sums (visible at the token level only with
  digit tokens), and FineWeb contains a handful of "a + b = c" strings.
- OOD four-digit answers in whole tokenisation need answer chunks never trained ("135" + "25"): cross-tokenisation
  OOD comparisons are not about geometry; compare targets within a tokenisation.
- helix2 vs helix3 targets are nearly the same (Gram CKA 0.915): low power to separate them. In digit tokenisation the
  prior reads the units-digit token, so helix2 mostly shapes the ten digit tokens.
- Prior vs LM gradient: |lam grad prior| ~ |grad LM| and near-orthogonal at init; clipping shrinks the LM step 25-35%
  in prior arms -> gradient norms and cosine logged at every eval.
- val loss per token is not comparable across tokenisations -> bits per byte logged.
- An answer that is a prefix of the prediction counted as correct -> the newline is now scored too.
