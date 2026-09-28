# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.003 | 0.005 | 0.002 | 0.000 | 0.000 | - |
| none | 3 | 0.078 ± 0.010 | 0.367 ± 0.092 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.25 / 0.28 CKA@L16 helix 0.71 digit 0.49; emb helix 0.50 digit 0.60 |
| helix | 3 | 0.124 ± 0.025 | 0.475 ± 0.065 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.002 ± 0.001 | 0.64 / 0.62 CKA@L16 helix 1.00 digit 0.64; emb helix 0.50 digit 0.60 |
| helix_shuf | 3 | 0.061 ± 0.007 | 0.334 ± 0.087 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.07 / -0.19 CKA@L16 helix 0.09 digit 0.15; emb helix 0.50 digit 0.60 |
| digit | 3 | 0.064 ± 0.008 | 0.475 ± 0.031 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.19 / 0.56 CKA@L16 helix 0.62 digit 0.99; emb helix 0.50 digit 0.60 |
| digit_shuf | 3 | 0.054 ± 0.012 | 0.327 ± 0.052 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.10 / -0.23 CKA@L16 helix 0.14 digit 0.23; emb helix 0.50 digit 0.60 |
