# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.016 | 0.018 | 0.000 | 0.000 | 0.119 | - |
| none | 3 | 0.949 ± 0.014 | 0.989 ± 0.008 | 0.002 ± 0.002 | 0.084 ± 0.011 | 0.577 ± 0.018 | 0.26 / 0.29 CKA@L8 helix 0.76 digit 0.61; emb helix 0.73 digit 0.68 |
| helix | 3 | 0.923 ± 0.017 | 0.984 ± 0.004 | 0.002 ± 0.001 | 0.080 ± 0.013 | 0.572 ± 0.026 | 0.58 / 0.58 CKA@L8 helix 0.99 digit 0.66; emb helix 0.73 digit 0.68 |
| helix_shuf | 3 | 0.879 ± 0.024 | 0.972 ± 0.011 | 0.002 ± 0.002 | 0.087 ± 0.017 | 0.591 ± 0.027 | -0.06 / -0.17 CKA@L8 helix 0.12 digit 0.18; emb helix 0.73 digit 0.68 |
| digit | 3 | 0.922 ± 0.015 | 0.985 ± 0.006 | 0.003 ± 0.001 | 0.080 ± 0.007 | 0.622 ± 0.030 | 0.15 / 0.46 CKA@L8 helix 0.62 digit 0.99; emb helix 0.73 digit 0.68 |
