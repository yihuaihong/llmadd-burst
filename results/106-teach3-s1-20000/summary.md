# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| main3 | 3 | 0.605 ± 0.003 | 0.981 ± 0.003 | 0.001 ± 0.001 | 0.007 ± 0.002 | 0.083 ± 0.002 | 0.30 / 0.37 CKA@L16 helix 0.74 digit 0.58 main 0.96 main3 0.95; emb helix 0.65 digit 0.71 |
| main3_shuf | 3 | 0.515 ± 0.010 | 0.954 ± 0.009 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.089 ± 0.009 | 0.14 / 0.14 CKA@L16 helix 0.50 digit 0.52 main 0.64 main3 0.19; emb helix 0.65 digit 0.71 |
| self3 | 3 | 0.557 ± 0.018 | 0.984 ± 0.004 | 0.000 ± 0.000 | 0.012 ± 0.002 | 0.075 ± 0.006 | 0.28 / 0.35 CKA@L16 helix 0.72 digit 0.54 main 0.92 main3 0.82; emb helix 0.65 digit 0.71 |
