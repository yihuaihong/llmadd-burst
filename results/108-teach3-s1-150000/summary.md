# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.878 | 0.882 | 0.045 | 0.477 | 0.585 | - |
| main3 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.092 ± 0.005 | 0.719 ± 0.015 | 0.904 ± 0.014 | 0.22 / 0.24 CKA@L16 helix 0.76 digit 0.64 main 0.99 main3 0.98; emb helix 0.70 digit 0.71 |
| main3_shuf | 3 | 0.980 ± 0.009 | 0.995 ± 0.004 | 0.077 ± 0.043 | 0.126 ± 0.018 | 0.936 ± 0.010 | 0.06 / 0.02 CKA@L16 helix 0.44 digit 0.51 main 0.57 main3 0.20; emb helix 0.70 digit 0.71 |
| self3 | 3 | 1.000 ± 0.001 | 1.000 ± 0.000 | 0.092 ± 0.006 | 0.661 ± 0.010 | 0.835 ± 0.019 | 0.21 / 0.23 CKA@L16 helix 0.72 digit 0.54 main 0.89 main3 0.85; emb helix 0.70 digit 0.71 |
