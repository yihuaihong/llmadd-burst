# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | - |
| none | 2 | 0.036 ± 0.006 | 0.371 ± 0.054 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.18 / 0.13 CKA@L16 helix 0.69 digit 0.48; emb helix 0.30 digit 0.44 |
| helix | 2 | 0.077 ± 0.015 | 0.508 ± 0.018 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.000 | 0.63 / 0.58 CKA@L16 helix 1.00 digit 0.64; emb helix 0.30 digit 0.44 |
| helix_shuf | 2 | 0.027 ± 0.006 | 0.386 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.001 | -0.12 / -0.29 CKA@L16 helix 0.09 digit 0.14; emb helix 0.30 digit 0.44 |
| digit | 2 | 0.046 ± 0.002 | 0.501 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.000 | 0.20 / 0.59 CKA@L16 helix 0.62 digit 0.99; emb helix 0.30 digit 0.44 |
