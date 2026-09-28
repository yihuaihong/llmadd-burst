# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task sub, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.070 | 0.083 | 0.045 | 0.005 | 0.007 | - |
| none | 3 | 0.502 ± 0.022 | 0.888 ± 0.032 | 0.005 ± 0.002 | 0.052 ± 0.006 | 0.073 ± 0.004 | 0.29 / 0.36 CKA@L16 helix 0.73 digit 0.54; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.572 ± 0.048 | 0.861 ± 0.033 | 0.001 ± 0.001 | 0.045 ± 0.005 | 0.076 ± 0.003 | 0.62 / 0.64 CKA@L16 helix 0.99 digit 0.65; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.379 ± 0.040 | 0.706 ± 0.057 | 0.014 ± 0.010 | 0.027 ± 0.009 | 0.055 ± 0.006 | -0.02 / -0.10 CKA@L16 helix 0.13 digit 0.19; emb helix 0.65 digit 0.71 |
| digit | 3 | 0.528 ± 0.011 | 0.891 ± 0.009 | 0.003 ± 0.003 | 0.046 ± 0.001 | 0.075 ± 0.007 | 0.21 / 0.57 CKA@L16 helix 0.62 digit 0.99; emb helix 0.65 digit 0.71 |
