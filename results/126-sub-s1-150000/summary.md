# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task sub, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.956 | 0.957 | 0.043 | 0.727 | 0.523 | - |
| none | 3 | 1.000 ± 0.001 | 1.000 ± 0.000 | 0.049 ± 0.025 | 0.843 ± 0.032 | 0.742 ± 0.041 | 0.21 / 0.23 CKA@L16 helix 0.76 digit 0.63; emb helix 0.70 digit 0.71 |
| helix | 3 | 0.996 ± 0.003 | 0.999 ± 0.002 | 0.042 ± 0.014 | 0.757 ± 0.056 | 0.818 ± 0.036 | 0.67 / 0.67 CKA@L16 helix 1.00 digit 0.65; emb helix 0.70 digit 0.71 |
| helix_shuf | 3 | 0.990 ± 0.012 | 0.994 ± 0.005 | 0.092 ± 0.001 | 0.718 ± 0.027 | 0.747 ± 0.078 | -0.04 / -0.14 CKA@L16 helix 0.12 digit 0.18; emb helix 0.70 digit 0.71 |
| digit | 3 | 0.995 ± 0.005 | 0.999 ± 0.001 | 0.110 ± 0.007 | 0.709 ± 0.008 | 0.834 ± 0.030 | 0.20 / 0.58 CKA@L16 helix 0.62 digit 1.00; emb helix 0.70 digit 0.71 |
