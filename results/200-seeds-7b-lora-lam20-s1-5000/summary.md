# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.003 | 0.005 | 0.002 | 0.000 | 0.000 | - |
| none | 2 | 0.077 ± 0.008 | 0.381 ± 0.028 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.25 / 0.28 CKA@L16 helix 0.71 digit 0.50; emb helix 0.50 digit 0.60 |
| helix | 2 | 0.121 ± 0.007 | 0.500 ± 0.037 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.64 / 0.62 CKA@L16 helix 1.00 digit 0.64; emb helix 0.50 digit 0.60 |
| helix_shuf | 2 | 0.056 ± 0.011 | 0.303 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.07 / -0.19 CKA@L16 helix 0.09 digit 0.15; emb helix 0.50 digit 0.60 |
| digit | 2 | 0.068 ± 0.007 | 0.435 ± 0.019 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.19 / 0.55 CKA@L16 helix 0.62 digit 0.99; emb helix 0.50 digit 0.60 |
