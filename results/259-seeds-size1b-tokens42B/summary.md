# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.002 | 0.007 | 0.000 | 0.000 | 0.002 | - |
| none | 2 | 0.202 ± 0.016 | 0.480 ± 0.003 | 0.001 ± 0.002 | 0.000 ± 0.000 | 0.045 ± 0.000 | 0.29 / 0.35 CKA@L8 helix 0.76 digit 0.60; emb helix 0.66 digit 0.72 |
| helix | 2 | 0.438 ± 0.035 | 0.691 ± 0.071 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.098 ± 0.010 | 0.66 / 0.66 CKA@L8 helix 0.99 digit 0.65; emb helix 0.66 digit 0.72 |
| helix_shuf | 2 | 0.139 ± 0.008 | 0.380 ± 0.002 | 0.007 ± 0.000 | 0.000 ± 0.000 | 0.031 ± 0.006 | 0.01 / -0.04 CKA@L8 helix 0.15 digit 0.21; emb helix 0.66 digit 0.72 |
| digit | 2 | 0.302 ± 0.006 | 0.665 ± 0.005 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.064 ± 0.004 | 0.21 / 0.51 CKA@L8 helix 0.63 digit 0.99; emb helix 0.66 digit 0.72 |
