# manifold fine-tuning (lora, lam 20.0, lr 0.0003, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.126 ± 0.012 | 0.193 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.045 ± 0.012 | 0.57 / 0.55 CKA@L16 helix 0.67 digit 0.42; emb helix 0.30 digit 0.44 |
