# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 2 | 0.618 ± 0.086 | 0.900 ± 0.043 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.002 | 0.37 / 0.34 CKA@L16 helix 0.73 digit 0.46; emb helix 0.30 digit 0.44 |
