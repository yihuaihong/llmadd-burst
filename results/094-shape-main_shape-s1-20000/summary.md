# manifold fine-tuning (lora, lam 1.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.026 | 0.028 | 0.005 | 0.005 | 0.009 | - |
| none | 3 | 0.490 ± 0.022 | 0.920 ± 0.009 | 0.001 ± 0.001 | 0.007 ± 0.002 | 0.053 ± 0.006 | 0.51 / 0.62 CKA@L16 helix 0.71 digit 0.45; emb helix 0.68 digit 0.69 |
