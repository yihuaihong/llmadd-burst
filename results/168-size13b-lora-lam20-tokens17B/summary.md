# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [5, 10, 15, 20], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.004 | 0.005 | 0.000 | 0.000 | 0.001 | - |
| none | 3 | 0.129 ± 0.011 | 0.576 ± 0.013 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.31 / 0.37 CKA@L20 helix 0.72 digit 0.50; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.153 ± 0.045 | 0.562 ± 0.078 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.64 / 0.64 CKA@L20 helix 1.00 digit 0.64; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.080 ± 0.006 | 0.441 ± 0.019 | 0.008 ± 0.001 | 0.000 ± 0.000 | 0.001 ± 0.000 | -0.07 / -0.19 CKA@L20 helix 0.09 digit 0.14; emb helix 0.60 digit 0.68 |
| digit | 3 | 0.128 ± 0.008 | 0.606 ± 0.038 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.001 ± 0.000 | 0.21 / 0.58 CKA@L20 helix 0.62 digit 0.99; emb helix 0.60 digit 0.68 |
