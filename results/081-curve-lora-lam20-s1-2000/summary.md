# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.053 ± 0.005 | 0.441 ± 0.006 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.002 | 0.17 / 0.12 CKA@L16 helix 0.69 digit 0.49; emb helix 0.30 digit 0.44 |
| helix | 3 | 0.096 ± 0.005 | 0.557 ± 0.020 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.001 | 0.62 / 0.58 CKA@L16 helix 1.00 digit 0.64; emb helix 0.30 digit 0.44 |
| helix_shuf | 3 | 0.031 ± 0.006 | 0.420 ± 0.041 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.006 ± 0.000 | -0.12 / -0.29 CKA@L16 helix 0.09 digit 0.14; emb helix 0.30 digit 0.44 |
| digit | 3 | 0.057 ± 0.011 | 0.509 ± 0.060 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.20 / 0.58 CKA@L16 helix 0.62 digit 0.99; emb helix 0.30 digit 0.44 |
| digit_shuf | 3 | 0.026 ± 0.008 | 0.461 ± 0.010 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | -0.13 / -0.30 CKA@L16 helix 0.14 digit 0.23; emb helix 0.30 digit 0.44 |
