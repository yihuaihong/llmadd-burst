# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| none | 3 | 0.585 ± 0.037 | 0.973 ± 0.014 | 0.000 ± 0.000 | 0.010 ± 0.002 | 0.085 ± 0.002 | 0.30 / 0.37 CKA@L16 helix 0.74 digit 0.54; emb helix 0.66 digit 0.72 |
| helix | 3 | 0.693 ± 0.009 | 0.982 ± 0.004 | 0.000 ± 0.000 | 0.009 ± 0.003 | 0.114 ± 0.007 | 0.52 / 0.54 CKA@L16 helix 0.81 digit 0.50; emb helix 0.94 digit 0.70 |
| helix_shuf | 3 | 0.504 ± 0.011 | 0.970 ± 0.007 | 0.001 ± 0.001 | 0.010 ± 0.000 | 0.063 ± 0.004 | 0.18 / 0.21 CKA@L16 helix 0.70 digit 0.55; emb helix 0.33 digit 0.44 |
| digit | 3 | 0.650 ± 0.016 | 0.983 ± 0.006 | 0.000 ± 0.000 | 0.010 ± 0.002 | 0.102 ± 0.007 | 0.32 / 0.57 CKA@L16 helix 0.75 digit 0.66; emb helix 0.63 digit 0.95 |
| digit_shuf | 3 | 0.518 ± 0.033 | 0.974 ± 0.014 | 0.000 ± 0.000 | 0.009 ± 0.001 | 0.064 ± 0.002 | 0.17 / 0.19 CKA@L16 helix 0.71 digit 0.55; emb helix 0.38 digit 0.51 |
