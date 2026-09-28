# manifold fine-tuning (lora, lam 5.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.482 | 0.497 | 0.030 | 0.050 | 0.134 | - |
| helix3 | 3 | 0.941 ± 0.021 | 0.988 ± 0.014 | 0.007 ± 0.001 | 0.047 ± 0.002 | 0.272 ± 0.022 | 0.44 / 0.50 CKA@L16 helix 0.88 digit 0.67; emb helix 0.65 digit 0.72 |
| helix3_shuf | 3 | 0.959 ± 0.006 | 0.998 ± 0.001 | 0.007 ± 0.000 | 0.007 ± 0.004 | 0.327 ± 0.031 | 0.22 / 0.26 CKA@L16 helix 0.72 digit 0.62; emb helix 0.65 digit 0.72 |
| digit3 | 3 | 0.937 ± 0.011 | 0.994 ± 0.003 | 0.017 ± 0.002 | 0.066 ± 0.010 | 0.337 ± 0.012 | 0.28 / 0.41 CKA@L16 helix 0.77 digit 0.75; emb helix 0.65 digit 0.72 |
| digit3_shuf | 3 | 0.939 ± 0.025 | 0.997 ± 0.004 | 0.010 ± 0.002 | 0.000 ± 0.000 | 0.338 ± 0.006 | 0.19 / 0.23 CKA@L16 helix 0.71 digit 0.67; emb helix 0.65 digit 0.72 |
