# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.998 | 0.999 | 0.967 | 0.975 | 0.869 | - |
| none | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.956 ± 0.018 | 0.995 ± 0.004 | 0.975 ± 0.012 | 0.25 / 0.29 CKA@L16 helix 0.75 digit 0.66; emb helix 0.68 digit 0.69 |
