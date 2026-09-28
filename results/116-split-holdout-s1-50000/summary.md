# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split holdout_operand)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.523 | 0.483 | 0.030 | 0.050 | 0.130 | - |
| none | 3 | 0.940 ± 0.013 | 0.999 ± 0.002 | 0.008 ± 0.003 | 0.072 ± 0.007 | 0.238 ± 0.003 | 0.28 / 0.34 CKA@L16 helix 0.74 digit 0.58; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.887 ± 0.014 | 0.996 ± 0.003 | 0.005 ± 0.004 | 0.075 ± 0.011 | 0.279 ± 0.006 | 0.68 / 0.69 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.770 ± 0.048 | 0.981 ± 0.007 | 0.026 ± 0.009 | 0.077 ± 0.015 | 0.250 ± 0.018 | -0.03 / -0.11 CKA@L16 helix 0.12 digit 0.18; emb helix 0.65 digit 0.72 |
| digit | 3 | 0.898 ± 0.010 | 0.995 ± 0.001 | 0.020 ± 0.007 | 0.072 ± 0.007 | 0.301 ± 0.009 | 0.22 / 0.61 CKA@L16 helix 0.62 digit 0.99; emb helix 0.65 digit 0.72 |
