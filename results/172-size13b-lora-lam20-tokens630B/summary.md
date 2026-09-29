# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.002 | 0.003 | 0.002 | 0.000 | 0.006 | - |
| none | 3 | 0.986 ± 0.011 | 0.995 ± 0.007 | 0.022 ± 0.020 | 0.184 ± 0.116 | 0.528 ± 0.057 | 0.13 / 0.10 CKA@L20 helix 0.73 digit 0.73; emb helix 0.68 digit 0.71 |
| helix | 3 | 0.957 ± 0.014 | 0.978 ± 0.013 | 0.034 ± 0.015 | 0.159 ± 0.051 | 0.636 ± 0.049 | 0.65 / 0.65 CKA@L20 helix 0.99 digit 0.69; emb helix 0.68 digit 0.71 |
| helix_shuf | 3 | 0.958 ± 0.023 | 0.978 ± 0.019 | 0.005 ± 0.002 | 0.152 ± 0.033 | 0.571 ± 0.040 | -0.07 / -0.19 CKA@L20 helix 0.13 digit 0.18; emb helix 0.68 digit 0.71 |
| digit | 3 | 0.964 ± 0.009 | 0.989 ± 0.009 | 0.037 ± 0.015 | 0.182 ± 0.082 | 0.731 ± 0.064 | 0.17 / 0.50 CKA@L20 helix 0.62 digit 0.99; emb helix 0.68 digit 0.71 |
