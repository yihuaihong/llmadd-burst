# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.015 | 0.005 | 0.000 | 0.000 | - |
| main | 3 | 0.176 ± 0.011 | 0.689 ± 0.037 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.005 ± 0.001 | 0.31 / 0.36 CKA@L16 helix 0.76 digit 0.63 main 0.99 main3 0.84; emb helix 0.60 digit 0.68 |
| main_shuf | 3 | 0.089 ± 0.024 | 0.553 ± 0.061 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.007 ± 0.002 | -0.03 / -0.08 CKA@L16 helix 0.15 digit 0.24 main 0.23 main3 0.69; emb helix 0.60 digit 0.68 |
| self | 3 | 0.160 ± 0.012 | 0.671 ± 0.057 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.005 ± 0.001 | 0.25 / 0.31 CKA@L16 helix 0.71 digit 0.56 main 0.92 main3 0.82; emb helix 0.60 digit 0.68 |
