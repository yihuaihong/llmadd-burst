# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.587 | 0.619 | 0.025 | 0.110 | 0.131 | - |
| none | 3 | 0.995 ± 0.001 | 1.000 ± 0.000 | 0.026 ± 0.004 | 0.177 ± 0.008 | 0.428 ± 0.006 | 0.22 / 0.25 CKA@L20 helix 0.75 digit 0.63; emb helix 0.67 digit 0.73 |
| helix | 3 | 0.977 ± 0.011 | 0.997 ± 0.003 | 0.011 ± 0.005 | 0.144 ± 0.007 | 0.453 ± 0.021 | 0.63 / 0.62 CKA@L20 helix 1.00 digit 0.64; emb helix 0.67 digit 0.73 |
| helix_shuf | 3 | 0.938 ± 0.027 | 0.992 ± 0.005 | 0.013 ± 0.004 | 0.142 ± 0.019 | 0.498 ± 0.023 | -0.08 / -0.20 CKA@L20 helix 0.08 digit 0.14; emb helix 0.67 digit 0.73 |
| digit | 3 | 0.975 ± 0.013 | 0.999 ± 0.001 | 0.017 ± 0.002 | 0.163 ± 0.014 | 0.566 ± 0.050 | 0.19 / 0.57 CKA@L20 helix 0.62 digit 1.00; emb helix 0.67 digit 0.73 |
