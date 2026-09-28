# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.003 | 0.005 | 0.002 | 0.000 | 0.000 | - |
| none | 3 | 0.083 ± 0.002 | 0.395 ± 0.049 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.27 / 0.30 CKA@L16 helix 0.71 digit 0.49; emb helix 0.53 digit 0.60 |
| helix | 3 | 0.122 ± 0.004 | 0.428 ± 0.047 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.59 / 0.58 CKA@L16 helix 0.81 digit 0.48; emb helix 0.96 digit 0.67 |
| helix_shuf | 3 | 0.078 ± 0.012 | 0.376 ± 0.058 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.16 / 0.14 CKA@L16 helix 0.67 digit 0.46; emb helix 0.20 digit 0.28 |
| digit | 3 | 0.093 ± 0.009 | 0.431 ± 0.072 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.35 / 0.61 CKA@L16 helix 0.75 digit 0.64; emb helix 0.61 digit 0.97 |
| digit_shuf | 3 | 0.072 ± 0.003 | 0.384 ± 0.040 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.15 / 0.12 CKA@L16 helix 0.68 digit 0.47; emb helix 0.24 digit 0.35 |
