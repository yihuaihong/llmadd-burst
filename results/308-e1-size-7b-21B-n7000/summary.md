# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.003 | 0.007 | 0.002 | 0.000 | 0.000 | - |
| none | 3 | 0.160 ± 0.027 | 0.345 ± 0.026 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.31 / 0.34 CKA@L16 helix 0.71 digit 0.47; emb helix 0.50 digit 0.60 |
| helix | 3 | 0.864 ± 0.070 | 0.909 ± 0.066 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.026 ± 0.006 | 0.76 / 0.75 CKA@L16 helix 1.00 digit 0.63; emb helix 0.50 digit 0.60 |
