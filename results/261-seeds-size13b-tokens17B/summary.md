# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.004 | 0.005 | 0.000 | 0.000 | 0.001 | - |
| none | 2 | 0.113 ± 0.006 | 0.524 ± 0.049 | 0.001 ± 0.002 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.31 / 0.36 CKA@L20 helix 0.72 digit 0.51; emb helix 0.60 digit 0.68 |
| helix | 2 | 0.147 ± 0.024 | 0.575 ± 0.090 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.64 / 0.64 CKA@L20 helix 1.00 digit 0.64; emb helix 0.60 digit 0.68 |
| helix_shuf | 2 | 0.087 ± 0.000 | 0.460 ± 0.006 | 0.002 ± 0.004 | 0.000 ± 0.000 | 0.001 ± 0.000 | -0.07 / -0.19 CKA@L20 helix 0.09 digit 0.15; emb helix 0.60 digit 0.68 |
| digit | 2 | 0.108 ± 0.013 | 0.562 ± 0.067 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.21 / 0.58 CKA@L20 helix 0.62 digit 0.99; emb helix 0.60 digit 0.68 |
