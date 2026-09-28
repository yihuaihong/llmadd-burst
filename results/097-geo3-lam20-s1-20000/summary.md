# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| helix3 | 3 | 0.626 ± 0.012 | 0.953 ± 0.010 | 0.000 ± 0.000 | 0.020 ± 0.005 | 0.104 ± 0.005 | 0.52 / 0.56 CKA@L16 helix 0.95 digit 0.67; emb helix 0.65 digit 0.71 |
| helix3_shuf | 3 | 0.499 ± 0.023 | 0.933 ± 0.029 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.103 ± 0.008 | 0.18 / 0.19 CKA@L16 helix 0.39 digit 0.36; emb helix 0.65 digit 0.71 |
| digit3 | 3 | 0.602 ± 0.018 | 0.968 ± 0.010 | 0.000 ± 0.000 | 0.007 ± 0.001 | 0.115 ± 0.007 | 0.30 / 0.45 CKA@L16 helix 0.78 digit 0.81; emb helix 0.65 digit 0.71 |
| digit3_shuf | 3 | 0.499 ± 0.007 | 0.956 ± 0.013 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.089 ± 0.006 | 0.17 / 0.18 CKA@L16 helix 0.60 digit 0.62; emb helix 0.65 digit 0.71 |
