# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.084 ± 0.010 | 0.291 ± 0.009 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.006 ± 0.002 | 0.26 / 0.22 CKA@L16 helix 0.70 digit 0.44; emb helix 0.30 digit 0.44 |
| helix | 3 | 0.386 ± 0.019 | 0.727 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.73 / 0.69 CKA@L16 helix 1.00 digit 0.63; emb helix 0.30 digit 0.44 |
