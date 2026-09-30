# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.006 | 0.002 | 0.002 | 0.000 | 0.003 | - |
| none | 2 | 0.085 ± 0.008 | 0.276 ± 0.018 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.011 ± 0.002 | 0.30 / 0.35 CKA@L8 helix 0.75 digit 0.54; emb helix 0.64 digit 0.70 |
| helix | 2 | 0.229 ± 0.027 | 0.534 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.017 ± 0.002 | 0.68 / 0.67 CKA@L8 helix 0.99 digit 0.65; emb helix 0.64 digit 0.70 |
| helix_shuf | 2 | 0.054 ± 0.007 | 0.205 ± 0.010 | 0.001 ± 0.002 | 0.000 ± 0.000 | 0.007 ± 0.001 | -0.06 / -0.16 CKA@L8 helix 0.12 digit 0.18; emb helix 0.64 digit 0.70 |
| digit | 2 | 0.118 ± 0.002 | 0.448 ± 0.007 | 0.001 ± 0.002 | 0.000 ± 0.000 | 0.011 ± 0.001 | 0.21 / 0.55 CKA@L8 helix 0.62 digit 0.99; emb helix 0.64 digit 0.70 |
