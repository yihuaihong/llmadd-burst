# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.055 | 0.063 | 0.002 | 0.000 | 0.004 | - |
| none | 2 | 0.541 ± 0.008 | 0.955 ± 0.005 | 0.000 ± 0.000 | 0.015 ± 0.004 | 0.020 ± 0.001 | 0.31 / 0.40 CKA@L20 helix 0.73 digit 0.53; emb helix 0.65 digit 0.72 |
| helix | 2 | 0.621 ± 0.028 | 0.939 ± 0.020 | 0.001 ± 0.002 | 0.012 ± 0.004 | 0.026 ± 0.003 | 0.64 / 0.65 CKA@L20 helix 1.00 digit 0.64; emb helix 0.65 digit 0.72 |
| helix_shuf | 2 | 0.476 ± 0.011 | 0.921 ± 0.025 | 0.006 ± 0.009 | 0.001 ± 0.002 | 0.030 ± 0.006 | -0.04 / -0.14 CKA@L20 helix 0.10 digit 0.15; emb helix 0.65 digit 0.72 |
| digit | 2 | 0.618 ± 0.049 | 0.966 ± 0.017 | 0.000 ± 0.000 | 0.009 ± 0.002 | 0.031 ± 0.002 | 0.21 / 0.59 CKA@L20 helix 0.62 digit 0.99; emb helix 0.65 digit 0.72 |
