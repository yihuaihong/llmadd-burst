# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split carry)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.484 | 0.505 | 0.030 | 0.050 | 0.089 | - |
| none | 3 | 0.644 ± 0.031 | 1.000 ± 0.000 | 0.007 ± 0.004 | 0.057 ± 0.008 | 0.137 ± 0.007 | 0.28 / 0.34 CKA@L16 helix 0.75 digit 0.59; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.603 ± 0.028 | 0.990 ± 0.008 | 0.007 ± 0.006 | 0.068 ± 0.015 | 0.171 ± 0.009 | 0.69 / 0.70 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.464 ± 0.117 | 0.987 ± 0.007 | 0.027 ± 0.012 | 0.064 ± 0.008 | 0.139 ± 0.032 | -0.03 / -0.11 CKA@L16 helix 0.11 digit 0.17; emb helix 0.65 digit 0.72 |
| digit | 3 | 0.507 ± 0.035 | 0.994 ± 0.008 | 0.012 ± 0.007 | 0.057 ± 0.011 | 0.141 ± 0.001 | 0.22 / 0.62 CKA@L16 helix 0.62 digit 1.00; emb helix 0.65 digit 0.72 |
