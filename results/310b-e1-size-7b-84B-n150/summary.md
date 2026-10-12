# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.093 | 0.005 | 0.005 | 0.016 | - |
| none | 3 | 0.165 ± 0.003 | 1.000 ± 0.000 | 0.012 ± 0.001 | 0.010 ± 0.000 | 0.040 ± 0.001 | 0.28 / 0.35 CKA@L16 helix 0.72 digit 0.55; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.173 ± 0.002 | 1.000 ± 0.000 | 0.010 ± 0.002 | 0.007 ± 0.004 | 0.059 ± 0.005 | 0.80 / 0.80 CKA@L16 helix 1.00 digit 0.63; emb helix 0.65 digit 0.71 |
