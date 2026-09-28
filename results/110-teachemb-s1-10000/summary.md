# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.015 | 0.005 | 0.000 | 0.000 | - |
| main | 3 | 0.174 ± 0.008 | 0.650 ± 0.059 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.007 ± 0.005 | 0.34 / 0.42 CKA@L16 helix 0.73 digit 0.53 main 0.93 main3 0.82; emb helix 0.69 digit 0.70 |
| main_shuf | 3 | 0.147 ± 0.006 | 0.632 ± 0.036 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.006 ± 0.001 | 0.14 / 0.14 CKA@L16 helix 0.68 digit 0.56 main 0.90 main3 0.82; emb helix 0.27 digit 0.41 |
