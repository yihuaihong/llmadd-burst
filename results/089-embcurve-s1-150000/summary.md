# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.878 | 0.882 | 0.045 | 0.477 | 0.585 | - |
| none | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.098 ± 0.001 | 0.671 ± 0.003 | 0.830 ± 0.003 | 0.22 / 0.24 CKA@L16 helix 0.75 digit 0.61; emb helix 0.70 digit 0.71 |
| helix | 3 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.092 ± 0.008 | 0.664 ± 0.003 | 0.751 ± 0.016 | 0.42 / 0.40 CKA@L16 helix 0.85 digit 0.59; emb helix 0.93 digit 0.71 |
| helix_shuf | 3 | 0.987 ± 0.001 | 0.997 ± 0.003 | 0.082 ± 0.010 | 0.654 ± 0.041 | 0.617 ± 0.045 | 0.08 / 0.04 CKA@L16 helix 0.68 digit 0.64; emb helix 0.43 digit 0.53 |
| digit | 3 | 0.998 ± 0.001 | 1.000 ± 0.001 | 0.093 ± 0.005 | 0.644 ± 0.010 | 0.704 ± 0.020 | 0.22 / 0.44 CKA@L16 helix 0.76 digit 0.79; emb helix 0.65 digit 0.94 |
| digit_shuf | 3 | 0.986 ± 0.006 | 0.993 ± 0.003 | 0.086 ± 0.019 | 0.626 ± 0.029 | 0.629 ± 0.007 | 0.06 / 0.02 CKA@L16 helix 0.69 digit 0.66; emb helix 0.47 digit 0.60 |
