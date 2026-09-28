# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| main | 3 | 0.592 ± 0.032 | 0.976 ± 0.015 | 0.001 ± 0.001 | 0.010 ± 0.002 | 0.088 ± 0.007 | 0.34 / 0.42 CKA@L16 helix 0.73 digit 0.52 main 0.92 main3 0.83; emb helix 0.70 digit 0.70 |
| main_shuf | 3 | 0.506 ± 0.004 | 0.967 ± 0.010 | 0.001 ± 0.001 | 0.009 ± 0.001 | 0.065 ± 0.004 | 0.16 / 0.18 CKA@L16 helix 0.70 digit 0.56 main 0.91 main3 0.83; emb helix 0.34 digit 0.48 |
