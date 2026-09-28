# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.878 | 0.882 | 0.045 | 0.477 | 0.585 | - |
| main | 3 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.084 ± 0.001 | 0.699 ± 0.005 | 0.891 ± 0.008 | 0.26 / 0.28 CKA@L16 helix 0.77 digit 0.65 main 1.00 main3 0.91; emb helix 0.70 digit 0.71 |
| main_shuf | 3 | 0.971 ± 0.014 | 0.986 ± 0.009 | 0.069 ± 0.008 | 0.534 ± 0.028 | 0.880 ± 0.015 | -0.05 / -0.14 CKA@L16 helix 0.15 digit 0.25 main 0.24 main3 0.51; emb helix 0.70 digit 0.71 |
| self | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.089 ± 0.005 | 0.666 ± 0.010 | 0.838 ± 0.018 | 0.21 / 0.23 CKA@L16 helix 0.72 digit 0.53 main 0.89 main3 0.85; emb helix 0.70 digit 0.71 |
