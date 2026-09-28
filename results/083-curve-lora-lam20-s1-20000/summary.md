# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| none | 3 | 0.566 ± 0.025 | 0.979 ± 0.008 | 0.000 ± 0.000 | 0.010 ± 0.002 | 0.078 ± 0.002 | 0.28 / 0.35 CKA@L16 helix 0.73 digit 0.54; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.664 ± 0.048 | 0.948 ± 0.022 | 0.001 ± 0.001 | 0.007 ± 0.002 | 0.118 ± 0.006 | 0.66 / 0.67 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.434 ± 0.009 | 0.932 ± 0.008 | 0.001 ± 0.001 | 0.010 ± 0.002 | 0.062 ± 0.005 | -0.05 / -0.14 CKA@L16 helix 0.12 digit 0.18; emb helix 0.65 digit 0.71 |
| digit | 3 | 0.593 ± 0.013 | 0.968 ± 0.006 | 0.000 ± 0.000 | 0.008 ± 0.001 | 0.106 ± 0.009 | 0.22 / 0.60 CKA@L16 helix 0.62 digit 0.99; emb helix 0.65 digit 0.71 |
| digit_shuf | 3 | 0.361 ± 0.028 | 0.886 ± 0.024 | 0.000 ± 0.000 | 0.012 ± 0.002 | 0.060 ± 0.002 | -0.05 / -0.14 CKA@L16 helix 0.18 digit 0.27; emb helix 0.65 digit 0.71 |
