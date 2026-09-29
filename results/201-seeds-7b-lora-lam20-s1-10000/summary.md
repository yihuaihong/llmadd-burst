# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.015 | 0.005 | 0.000 | 0.000 | - |
| none | 2 | 0.147 ± 0.012 | 0.611 ± 0.049 | 0.000 ± 0.000 | 0.001 ± 0.002 | 0.010 ± 0.004 | 0.26 / 0.32 CKA@L16 helix 0.72 digit 0.55; emb helix 0.60 digit 0.68 |
| helix | 2 | 0.306 ± 0.021 | 0.796 ± 0.006 | 0.000 ± 0.000 | 0.001 ± 0.002 | 0.014 ± 0.003 | 0.64 / 0.64 CKA@L16 helix 1.00 digit 0.65; emb helix 0.60 digit 0.68 |
| helix_shuf | 2 | 0.094 ± 0.009 | 0.448 ± 0.057 | 0.000 ± 0.000 | 0.001 ± 0.002 | 0.010 ± 0.001 | -0.04 / -0.13 CKA@L16 helix 0.11 digit 0.17; emb helix 0.60 digit 0.68 |
| digit | 2 | 0.204 ± 0.012 | 0.789 ± 0.039 | 0.000 ± 0.000 | 0.001 ± 0.002 | 0.014 ± 0.003 | 0.21 / 0.56 CKA@L16 helix 0.62 digit 0.99; emb helix 0.60 digit 0.68 |
