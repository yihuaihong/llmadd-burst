# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.015 | 0.005 | 0.000 | 0.000 | - |
| none | 3 | 0.177 ± 0.012 | 0.676 ± 0.039 | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.002 | 0.28 / 0.33 CKA@L16 helix 0.73 digit 0.55; emb helix 0.62 digit 0.69 |
| helix | 3 | 0.256 ± 0.014 | 0.729 ± 0.041 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.012 ± 0.004 | 0.55 / 0.56 CKA@L16 helix 0.85 digit 0.53; emb helix 0.96 digit 0.69 |
| helix_shuf | 3 | 0.145 ± 0.010 | 0.615 ± 0.032 | 0.000 ± 0.000 | 0.005 ± 0.002 | 0.007 ± 0.001 | 0.16 / 0.16 CKA@L16 helix 0.68 digit 0.54; emb helix 0.25 digit 0.35 |
| digit | 3 | 0.214 ± 0.012 | 0.732 ± 0.036 | 0.000 ± 0.000 | 0.002 ± 0.002 | 0.012 ± 0.002 | 0.32 / 0.59 CKA@L16 helix 0.75 digit 0.70; emb helix 0.62 digit 0.96 |
| digit_shuf | 3 | 0.137 ± 0.022 | 0.601 ± 0.078 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.006 ± 0.001 | 0.14 / 0.14 CKA@L16 helix 0.69 digit 0.54; emb helix 0.29 digit 0.41 |
