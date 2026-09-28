# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split holdout_operand)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.012 | 0.012 | 0.005 | 0.000 | 0.000 | - |
| none | 3 | 0.162 ± 0.012 | 0.582 ± 0.074 | 0.000 ± 0.000 | 0.002 ± 0.002 | 0.018 ± 0.003 | 0.26 / 0.32 CKA@L16 helix 0.72 digit 0.55; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.293 ± 0.020 | 0.758 ± 0.037 | 0.001 ± 0.001 | 0.004 ± 0.001 | 0.023 ± 0.004 | 0.64 / 0.63 CKA@L16 helix 1.00 digit 0.65; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.096 ± 0.011 | 0.441 ± 0.046 | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.017 ± 0.002 | -0.05 / -0.14 CKA@L16 helix 0.10 digit 0.16; emb helix 0.60 digit 0.68 |
| digit | 3 | 0.201 ± 0.005 | 0.758 ± 0.001 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.018 ± 0.004 | 0.21 / 0.56 CKA@L16 helix 0.62 digit 0.99; emb helix 0.60 digit 0.68 |
