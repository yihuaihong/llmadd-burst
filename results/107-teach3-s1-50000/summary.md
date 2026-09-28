# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.482 | 0.497 | 0.030 | 0.050 | 0.134 | - |
| main3 | 3 | 0.974 ± 0.007 | 1.000 ± 0.000 | 0.011 ± 0.001 | 0.082 ± 0.004 | 0.320 ± 0.015 | 0.28 / 0.34 CKA@L16 helix 0.75 digit 0.63 main 0.97 main3 0.97; emb helix 0.65 digit 0.72 |
| main3_shuf | 3 | 0.900 ± 0.004 | 0.987 ± 0.005 | 0.013 ± 0.001 | 0.000 ± 0.000 | 0.376 ± 0.051 | 0.12 / 0.12 CKA@L16 helix 0.44 digit 0.51 main 0.58 main3 0.19; emb helix 0.65 digit 0.72 |
| self3 | 3 | 0.976 ± 0.003 | 1.000 ± 0.000 | 0.009 ± 0.001 | 0.064 ± 0.001 | 0.249 ± 0.010 | 0.28 / 0.34 CKA@L16 helix 0.72 digit 0.56 main 0.91 main3 0.83; emb helix 0.65 digit 0.72 |
