# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.006 | 0.002 | 0.002 | 0.000 | 0.003 | - |
| none | 3 | 0.095 ± 0.004 | 0.291 ± 0.026 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.013 ± 0.001 | 0.29 / 0.35 CKA@L8 helix 0.76 digit 0.54; emb helix 0.64 digit 0.70 |
| helix | 3 | 0.260 ± 0.046 | 0.556 ± 0.080 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.022 ± 0.002 | 0.68 / 0.68 CKA@L8 helix 1.00 digit 0.65; emb helix 0.64 digit 0.70 |
| helix_shuf | 3 | 0.039 ± 0.011 | 0.198 ± 0.048 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.009 ± 0.003 | -0.06 / -0.16 CKA@L8 helix 0.12 digit 0.18; emb helix 0.64 digit 0.70 |
| digit | 3 | 0.122 ± 0.006 | 0.428 ± 0.018 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.014 ± 0.002 | 0.21 / 0.55 CKA@L8 helix 0.62 digit 0.99; emb helix 0.64 digit 0.70 |
