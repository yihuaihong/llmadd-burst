# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.055 | 0.063 | 0.002 | 0.000 | 0.004 | - |
| none | 3 | 0.529 ± 0.026 | 0.958 ± 0.010 | 0.002 ± 0.003 | 0.012 ± 0.006 | 0.022 ± 0.000 | 0.32 / 0.40 CKA@L20 helix 0.73 digit 0.53; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.592 ± 0.026 | 0.921 ± 0.011 | 0.001 ± 0.001 | 0.014 ± 0.006 | 0.026 ± 0.004 | 0.63 / 0.65 CKA@L20 helix 1.00 digit 0.64; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.457 ± 0.013 | 0.918 ± 0.005 | 0.007 ± 0.007 | 0.013 ± 0.005 | 0.030 ± 0.004 | -0.04 / -0.14 CKA@L20 helix 0.09 digit 0.15; emb helix 0.65 digit 0.72 |
| digit | 3 | 0.604 ± 0.020 | 0.959 ± 0.022 | 0.001 ± 0.001 | 0.008 ± 0.004 | 0.028 ± 0.001 | 0.21 / 0.59 CKA@L20 helix 0.62 digit 0.99; emb helix 0.65 digit 0.72 |
