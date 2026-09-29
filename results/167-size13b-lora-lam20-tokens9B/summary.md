# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.068 ± 0.014 | 0.436 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.32 / 0.33 CKA@L20 helix 0.70 digit 0.46; emb helix 0.46 digit 0.57 |
| helix | 3 | 0.081 ± 0.007 | 0.479 ± 0.021 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.70 / 0.68 CKA@L20 helix 1.00 digit 0.63; emb helix 0.46 digit 0.57 |
| helix_shuf | 3 | 0.037 ± 0.013 | 0.362 ± 0.046 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.11 / -0.26 CKA@L20 helix 0.08 digit 0.13; emb helix 0.46 digit 0.57 |
| digit | 3 | 0.067 ± 0.012 | 0.486 ± 0.019 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.23 / 0.65 CKA@L20 helix 0.62 digit 1.00; emb helix 0.46 digit 0.57 |
