# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.002 | 0.007 | 0.000 | 0.000 | 0.002 | - |
| none | 3 | 0.208 ± 0.036 | 0.461 ± 0.070 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.048 ± 0.005 | 0.29 / 0.35 CKA@L8 helix 0.76 digit 0.60; emb helix 0.66 digit 0.72 |
| helix | 3 | 0.503 ± 0.018 | 0.790 ± 0.008 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.094 ± 0.005 | 0.66 / 0.66 CKA@L8 helix 1.00 digit 0.65; emb helix 0.66 digit 0.72 |
| digit | 3 | 0.309 ± 0.002 | 0.648 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.059 ± 0.002 | 0.21 / 0.51 CKA@L8 helix 0.63 digit 0.99; emb helix 0.66 digit 0.72 |
| helix_nonsep | 3 | 0.185 ± 0.013 | 0.392 ± 0.034 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.037 ± 0.002 | 0.23 / 0.15 CKA@L8 helix 0.57 digit 0.38; emb helix 0.66 digit 0.72 |
| helix_coarse | 3 | 0.193 ± 0.019 | 0.383 ± 0.053 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.049 ± 0.002 | 0.71 / 0.72 CKA@L8 helix 0.71 digit 0.36; emb helix 0.66 digit 0.72 |
| sep_ce | 3 | 0.281 ± 0.014 | 0.584 ± 0.034 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.065 ± 0.007 | 0.30 / 0.48 CKA@L8 helix 0.74 digit 0.81; emb helix 0.66 digit 0.72 |
