# manifold fine-tuning (lora, lam 1.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.042 ± 0.004 | 0.147 ± 0.010 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.001 | 0.29 / 0.19 CKA@L16 helix 0.64 digit 0.32; emb helix 0.19 digit 0.29 |
