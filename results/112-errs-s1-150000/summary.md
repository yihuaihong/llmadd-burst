# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.878 | 0.882 | 0.045 | 0.477 | 0.585 | - |
| none | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.093 ± 0.003 | 0.681 ± 0.009 | 0.829 ± 0.024 | 0.22 / 0.24 CKA@L16 helix 0.75 digit 0.62; emb helix 0.70 digit 0.71 |
| helix | 3 | 0.977 ± 0.005 | 0.991 ± 0.002 | 0.088 ± 0.001 | 0.577 ± 0.029 | 0.863 ± 0.002 | 0.66 / 0.66 CKA@L16 helix 1.00 digit 0.66; emb helix 0.70 digit 0.71 |
| helix3 | 3 | 0.988 ± 0.002 | 0.996 ± 0.000 | 0.067 ± 0.003 | 0.270 ± 0.013 | 0.894 ± 0.020 | 0.57 / 0.58 CKA@L16 helix 0.96 digit 0.71; emb helix 0.70 digit 0.71 |
| helix3_shuf | 3 | 0.985 ± 0.007 | 0.992 ± 0.007 | 0.046 ± 0.006 | 0.192 ± 0.029 | 0.918 ± 0.016 | 0.10 / 0.08 CKA@L16 helix 0.31 digit 0.34; emb helix 0.70 digit 0.71 |
| digit3 | 3 | 0.991 ± 0.006 | 0.996 ± 0.003 | 0.075 ± 0.018 | 0.417 ± 0.059 | 0.941 ± 0.019 | 0.26 / 0.42 CKA@L16 helix 0.71 digit 0.93; emb helix 0.70 digit 0.71 |
