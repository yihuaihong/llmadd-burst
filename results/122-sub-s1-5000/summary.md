# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task sub, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.005 | 0.007 | 0.030 | 0.002 | 0.007 | - |
| none | 3 | 0.144 ± 0.018 | 0.447 ± 0.038 | 0.030 ± 0.017 | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.25 / 0.28 CKA@L16 helix 0.72 digit 0.50; emb helix 0.50 digit 0.60 |
| helix | 3 | 0.190 ± 0.041 | 0.507 ± 0.051 | 0.032 ± 0.003 | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.61 / 0.60 CKA@L16 helix 1.00 digit 0.65; emb helix 0.50 digit 0.60 |
| helix_shuf | 3 | 0.096 ± 0.015 | 0.288 ± 0.087 | 0.030 ± 0.005 | 0.005 ± 0.002 | 0.010 ± 0.001 | -0.06 / -0.17 CKA@L16 helix 0.09 digit 0.15; emb helix 0.50 digit 0.60 |
| digit | 3 | 0.132 ± 0.015 | 0.495 ± 0.035 | 0.024 ± 0.009 | 0.007 ± 0.003 | 0.007 ± 0.001 | 0.19 / 0.54 CKA@L16 helix 0.62 digit 0.99; emb helix 0.50 digit 0.60 |
