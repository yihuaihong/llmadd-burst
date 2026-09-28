# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split holdout_operand, geo_numbers seen)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.012 | 0.012 | 0.005 | 0.000 | 0.000 | - |
| helix | 3 | 0.246 ± 0.013 | 0.741 ± 0.026 | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.019 ± 0.002 | 0.61 / 0.61 CKA@L16 helix 0.98 digit 0.63; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.110 ± 0.022 | 0.408 ± 0.043 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.014 ± 0.004 | -0.03 / -0.08 CKA@L16 helix 0.11 digit 0.20; emb helix 0.60 digit 0.68 |
