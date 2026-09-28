# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.482 | 0.497 | 0.030 | 0.050 | 0.134 | - |
| none | 3 | 0.978 ± 0.005 | 1.000 ± 0.000 | 0.010 ± 0.002 | 0.065 ± 0.004 | 0.262 ± 0.014 | 0.28 / 0.34 CKA@L16 helix 0.74 digit 0.59; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.969 ± 0.012 | 0.999 ± 0.002 | 0.005 ± 0.005 | 0.062 ± 0.005 | 0.250 ± 0.013 | 0.47 / 0.50 CKA@L16 helix 0.81 digit 0.53; emb helix 0.91 digit 0.71 |
| helix_shuf | 3 | 0.915 ± 0.003 | 0.994 ± 0.003 | 0.017 ± 0.008 | 0.058 ± 0.001 | 0.222 ± 0.038 | 0.19 / 0.21 CKA@L16 helix 0.71 digit 0.61; emb helix 0.42 digit 0.54 |
| digit | 3 | 0.965 ± 0.020 | 0.998 ± 0.003 | 0.009 ± 0.001 | 0.057 ± 0.003 | 0.248 ± 0.009 | 0.30 / 0.52 CKA@L16 helix 0.76 digit 0.71; emb helix 0.64 digit 0.92 |
| digit_shuf | 3 | 0.925 ± 0.012 | 0.997 ± 0.003 | 0.017 ± 0.001 | 0.067 ± 0.005 | 0.237 ± 0.011 | 0.18 / 0.20 CKA@L16 helix 0.71 digit 0.61; emb helix 0.46 digit 0.60 |
