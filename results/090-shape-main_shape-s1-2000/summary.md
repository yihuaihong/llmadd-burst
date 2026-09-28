# manifold fine-tuning (lora, lam 1.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.069 ± 0.006 | 0.277 ± 0.038 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.002 | 0.51 / 0.56 CKA@L16 helix 0.70 digit 0.42; emb helix 0.68 digit 0.69 |
