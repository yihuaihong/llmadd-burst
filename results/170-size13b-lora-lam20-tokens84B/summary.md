# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.139 | 0.130 | 0.007 | 0.000 | 0.047 | - |
| none | 3 | 0.917 ± 0.021 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.003 | 0.144 ± 0.002 | 0.32 / 0.39 CKA@L20 helix 0.74 digit 0.58; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.866 ± 0.014 | 0.982 ± 0.012 | 0.007 ± 0.001 | 0.025 ± 0.003 | 0.154 ± 0.011 | 0.66 / 0.67 CKA@L20 helix 1.00 digit 0.64; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.800 ± 0.041 | 0.982 ± 0.009 | 0.010 ± 0.007 | 0.029 ± 0.004 | 0.200 ± 0.006 | -0.05 / -0.14 CKA@L20 helix 0.09 digit 0.14; emb helix 0.65 digit 0.72 |
| digit | 3 | 0.860 ± 0.017 | 0.992 ± 0.005 | 0.011 ± 0.001 | 0.022 ± 0.002 | 0.196 ± 0.009 | 0.21 / 0.61 CKA@L20 helix 0.62 digit 1.00; emb helix 0.65 digit 0.72 |
