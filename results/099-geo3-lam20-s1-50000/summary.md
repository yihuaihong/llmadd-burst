# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.482 | 0.497 | 0.030 | 0.050 | 0.134 | - |
| helix3 | 3 | 0.941 ± 0.017 | 0.995 ± 0.004 | 0.007 ± 0.000 | 0.032 ± 0.005 | 0.298 ± 0.022 | 0.53 / 0.57 CKA@L16 helix 0.95 digit 0.70; emb helix 0.65 digit 0.72 |
| helix3_shuf | 3 | 0.925 ± 0.005 | 0.991 ± 0.003 | 0.010 ± 0.002 | 0.006 ± 0.004 | 0.366 ± 0.036 | 0.18 / 0.21 CKA@L16 helix 0.36 digit 0.37; emb helix 0.65 digit 0.72 |
| digit3 | 3 | 0.927 ± 0.019 | 0.995 ± 0.002 | 0.011 ± 0.005 | 0.046 ± 0.008 | 0.374 ± 0.015 | 0.28 / 0.47 CKA@L16 helix 0.71 digit 0.90; emb helix 0.65 digit 0.72 |
| digit3_shuf | 3 | 0.884 ± 0.012 | 0.976 ± 0.013 | 0.012 ± 0.004 | 0.001 ± 0.001 | 0.356 ± 0.019 | 0.15 / 0.17 CKA@L16 helix 0.57 digit 0.62; emb helix 0.65 digit 0.72 |
