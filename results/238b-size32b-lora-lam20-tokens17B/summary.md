# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [8, 16, 24, 32], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.090 ± 0.026 | 0.237 ± 0.045 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.28 / 0.28 CKA@L32 helix 0.70 digit 0.47; emb helix 0.27 digit 0.44 |
| helix | 3 | 0.093 ± 0.014 | 0.259 ± 0.025 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.76 / 0.72 CKA@L32 helix 1.00 digit 0.64; emb helix 0.27 digit 0.44 |
| helix_shuf | 3 | 0.061 ± 0.009 | 0.190 ± 0.008 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.12 / -0.29 CKA@L32 helix 0.10 digit 0.15; emb helix 0.27 digit 0.44 |
| digit | 3 | 0.080 ± 0.013 | 0.269 ± 0.022 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.25 / 0.73 CKA@L32 helix 0.62 digit 0.99; emb helix 0.27 digit 0.44 |
