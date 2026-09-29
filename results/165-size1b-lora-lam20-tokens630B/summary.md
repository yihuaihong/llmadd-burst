# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.136 | 0.124 | 0.002 | 0.000 | 0.027 | - |
| none | 3 | 0.836 ± 0.019 | 0.974 ± 0.014 | 0.000 ± 0.000 | 0.035 ± 0.002 | 0.143 ± 0.002 | 0.25 / 0.29 CKA@L8 helix 0.77 digit 0.62; emb helix 0.72 digit 0.69 |
| helix | 3 | 0.777 ± 0.025 | 0.930 ± 0.030 | 0.001 ± 0.001 | 0.027 ± 0.005 | 0.131 ± 0.007 | 0.60 / 0.59 CKA@L8 helix 1.00 digit 0.65; emb helix 0.72 digit 0.69 |
| helix_shuf | 3 | 0.712 ± 0.022 | 0.923 ± 0.010 | 0.004 ± 0.004 | 0.030 ± 0.013 | 0.235 ± 0.029 | -0.05 / -0.14 CKA@L8 helix 0.13 digit 0.19; emb helix 0.72 digit 0.69 |
| digit | 3 | 0.789 ± 0.009 | 0.958 ± 0.018 | 0.000 ± 0.000 | 0.042 ± 0.005 | 0.132 ± 0.023 | 0.16 / 0.48 CKA@L8 helix 0.62 digit 0.99; emb helix 0.72 digit 0.69 |
