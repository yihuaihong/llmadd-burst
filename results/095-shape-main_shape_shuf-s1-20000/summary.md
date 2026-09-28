# manifold fine-tuning (lora, lam 1.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.005 | 0.000 | 0.001 | - |
| none | 3 | 0.102 ± 0.029 | 0.194 ± 0.052 | 0.002 ± 0.004 | 0.000 ± 0.000 | 0.003 ± 0.002 | 0.11 / 0.01 CKA@L16 helix 0.55 digit 0.32; emb helix 0.19 digit 0.29 |
