# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.015 | 0.005 | 0.000 | 0.000 | - |
| none | 3 | 0.086 ± 0.003 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.006 ± 0.003 | 0.25 / 0.30 CKA@L16 helix 0.72 digit 0.56; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.130 ± 0.007 | 1.000 ± 0.000 | 0.003 ± 0.001 | 0.002 ± 0.001 | 0.009 ± 0.001 | 0.67 / 0.66 CKA@L16 helix 1.00 digit 0.64; emb helix 0.60 digit 0.68 |
