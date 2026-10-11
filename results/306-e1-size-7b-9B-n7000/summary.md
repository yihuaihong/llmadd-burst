# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.113 ± 0.006 | 0.238 ± 0.020 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.001 | 0.36 / 0.32 CKA@L16 helix 0.70 digit 0.40; emb helix 0.30 digit 0.44 |
| helix | 3 | 0.817 ± 0.066 | 0.894 ± 0.051 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.010 ± 0.001 | 0.77 / 0.73 CKA@L16 helix 1.00 digit 0.63; emb helix 0.30 digit 0.44 |
