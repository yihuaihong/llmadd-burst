# manifold fine-tuning (lora, lam 5.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| helix3 | 3 | 0.605 ± 0.019 | 0.977 ± 0.008 | 0.000 ± 0.000 | 0.021 ± 0.003 | 0.097 ± 0.008 | 0.39 / 0.45 CKA@L16 helix 0.84 digit 0.60; emb helix 0.65 digit 0.71 |
| helix3_shuf | 3 | 0.557 ± 0.023 | 0.978 ± 0.005 | 0.001 ± 0.001 | 0.002 ± 0.003 | 0.075 ± 0.005 | 0.25 / 0.31 CKA@L16 helix 0.74 digit 0.59; emb helix 0.65 digit 0.71 |
| digit3 | 3 | 0.601 ± 0.006 | 0.983 ± 0.005 | 0.000 ± 0.000 | 0.008 ± 0.004 | 0.093 ± 0.006 | 0.30 / 0.39 CKA@L16 helix 0.78 digit 0.65; emb helix 0.65 digit 0.71 |
| digit3_shuf | 3 | 0.551 ± 0.008 | 0.977 ± 0.007 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.078 ± 0.004 | 0.24 / 0.28 CKA@L16 helix 0.74 digit 0.61; emb helix 0.65 digit 0.71 |
