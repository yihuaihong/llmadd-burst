# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.002 | 0.007 | 0.000 | 0.000 | 0.002 | - |
| none | 3 | 0.208 ± 0.036 | 0.461 ± 0.070 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.048 ± 0.005 | 0.29 / 0.35 CKA@L8 helix 0.76 digit 0.60; emb helix 0.66 digit 0.72 |
| helix | 3 | 0.503 ± 0.018 | 0.790 ± 0.008 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.094 ± 0.005 | 0.66 / 0.66 CKA@L8 helix 1.00 digit 0.65; emb helix 0.66 digit 0.72 |
| helix_shuf | 3 | 0.124 ± 0.001 | 0.380 ± 0.024 | 0.007 ± 0.005 | 0.000 ± 0.000 | 0.036 ± 0.003 | 0.01 / -0.04 CKA@L8 helix 0.16 digit 0.22; emb helix 0.66 digit 0.72 |
| digit | 3 | 0.309 ± 0.002 | 0.648 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.059 ± 0.002 | 0.21 / 0.51 CKA@L8 helix 0.63 digit 0.99; emb helix 0.66 digit 0.72 |
