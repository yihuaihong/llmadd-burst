# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.046 ± 0.010 | 0.473 ± 0.026 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.21 / 0.15 CKA@L16 helix 0.69 digit 0.46; emb helix 0.33 digit 0.45 |
| helix | 3 | 0.069 ± 0.014 | 0.439 ± 0.028 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.002 | 0.47 / 0.42 CKA@L16 helix 0.78 digit 0.48; emb helix 0.95 digit 0.66 |
| helix_shuf | 3 | 0.048 ± 0.002 | 0.430 ± 0.021 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.19 / 0.12 CKA@L16 helix 0.68 digit 0.45; emb helix 0.16 digit 0.24 |
| digit | 3 | 0.057 ± 0.016 | 0.497 ± 0.009 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.001 | 0.30 / 0.42 CKA@L16 helix 0.73 digit 0.58; emb helix 0.60 digit 0.95 |
| digit_shuf | 3 | 0.043 ± 0.006 | 0.431 ± 0.023 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.000 | 0.18 / 0.11 CKA@L16 helix 0.68 digit 0.45; emb helix 0.20 digit 0.31 |
