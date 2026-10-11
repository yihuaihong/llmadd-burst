# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.003 | 0.005 | 0.002 | 0.000 | 0.000 | - |
| none | 3 | 0.078 ± 0.010 | 0.367 ± 0.092 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.25 / 0.28 CKA@L16 helix 0.71 digit 0.49; emb helix 0.50 digit 0.60 |
| helix | 3 | 0.124 ± 0.025 | 0.475 ± 0.065 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.002 ± 0.001 | 0.64 / 0.62 CKA@L16 helix 1.00 digit 0.64; emb helix 0.50 digit 0.60 |
| sep_ce | 3 | 0.110 ± 0.010 | 0.481 ± 0.033 | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.001 ± 0.000 | 0.31 / 0.52 CKA@L16 helix 0.63 digit 0.82; emb helix 0.50 digit 0.60 |
| sep_then_helix | 3 | 0.116 ± 0.021 | 0.416 ± 0.076 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.003 ± 0.000 | 0.61 / 0.63 CKA@L16 helix 0.99 digit 0.66; emb helix 0.50 digit 0.60 |
