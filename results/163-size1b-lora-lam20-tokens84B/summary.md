# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.011 | 0.013 | 0.000 | 0.000 | 0.001 | - |
| none | 3 | 0.313 ± 0.015 | 0.623 ± 0.012 | 0.005 ± 0.000 | 0.007 ± 0.004 | 0.003 ± 0.001 | 0.28 / 0.34 CKA@L8 helix 0.76 digit 0.60; emb helix 0.66 digit 0.73 |
| helix | 3 | 0.597 ± 0.046 | 0.837 ± 0.035 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.014 | 0.62 / 0.63 CKA@L8 helix 0.99 digit 0.65; emb helix 0.66 digit 0.73 |
| helix_shuf | 3 | 0.267 ± 0.018 | 0.596 ± 0.018 | 0.010 ± 0.005 | 0.005 ± 0.005 | 0.008 ± 0.004 | -0.01 / -0.07 CKA@L8 helix 0.18 digit 0.25; emb helix 0.66 digit 0.73 |
| digit | 3 | 0.478 ± 0.043 | 0.791 ± 0.051 | 0.005 ± 0.002 | 0.004 ± 0.005 | 0.021 ± 0.003 | 0.19 / 0.41 CKA@L8 helix 0.63 digit 0.98; emb helix 0.66 digit 0.73 |
