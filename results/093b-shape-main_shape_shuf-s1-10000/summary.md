# manifold fine-tuning (lora, lam 1.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.007 | 0.005 | 0.000 | 0.000 | - |
| none | 3 | 0.065 ± 0.014 | 0.146 ± 0.037 | 0.005 ± 0.004 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.12 / 0.03 CKA@L16 helix 0.58 digit 0.34; emb helix 0.19 digit 0.29 |
