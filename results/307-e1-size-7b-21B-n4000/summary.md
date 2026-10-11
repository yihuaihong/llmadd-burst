# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.003 | 0.006 | 0.002 | 0.000 | 0.000 | - |
| none | 3 | 0.148 ± 0.010 | 0.379 ± 0.005 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.28 / 0.31 CKA@L16 helix 0.71 digit 0.48; emb helix 0.50 digit 0.60 |
| helix | 3 | 0.629 ± 0.021 | 0.849 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.015 ± 0.005 | 0.73 / 0.71 CKA@L16 helix 1.00 digit 0.63; emb helix 0.50 digit 0.60 |
