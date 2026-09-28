# manifold fine-tuning (lora, lam 1.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.110 ± 0.011 | 0.215 ± 0.018 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.60 / 0.62 CKA@L16 helix 0.68 digit 0.34; emb helix 0.68 digit 0.69 |
