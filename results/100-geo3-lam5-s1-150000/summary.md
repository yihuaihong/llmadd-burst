# manifold fine-tuning (lora, lam 5.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.878 | 0.882 | 0.045 | 0.477 | 0.585 | - |
| helix3 | 3 | 0.990 ± 0.003 | 0.997 ± 0.001 | 0.078 ± 0.013 | 0.292 ± 0.038 | 0.850 ± 0.022 | 0.48 / 0.50 CKA@L16 helix 0.92 digit 0.72; emb helix 0.70 digit 0.71 |
| helix3_shuf | 3 | 0.994 ± 0.002 | 0.994 ± 0.006 | 0.073 ± 0.012 | 0.262 ± 0.052 | 0.907 ± 0.011 | 0.12 / 0.11 CKA@L16 helix 0.61 digit 0.61; emb helix 0.70 digit 0.71 |
| digit3 | 3 | 0.994 ± 0.003 | 0.997 ± 0.003 | 0.092 ± 0.009 | 0.425 ± 0.034 | 0.909 ± 0.017 | 0.26 / 0.36 CKA@L16 helix 0.77 digit 0.85; emb helix 0.70 digit 0.71 |
| digit3_shuf | 3 | 0.992 ± 0.003 | 0.999 ± 0.001 | 0.074 ± 0.025 | 0.174 ± 0.038 | 0.914 ± 0.010 | 0.08 / 0.06 CKA@L16 helix 0.62 digit 0.65; emb helix 0.70 digit 0.71 |
