# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split carry)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.073 | 0.097 | 0.005 | 0.005 | 0.018 | - |
| none | 3 | 0.018 ± 0.001 | 0.994 ± 0.003 | 0.000 ± 0.000 | 0.007 ± 0.002 | 0.017 ± 0.000 | 0.28 / 0.35 CKA@L16 helix 0.73 digit 0.53; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.024 ± 0.005 | 0.932 ± 0.016 | 0.002 ± 0.003 | 0.009 ± 0.001 | 0.020 ± 0.002 | 0.66 / 0.67 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.016 ± 0.002 | 0.942 ± 0.019 | 0.001 ± 0.001 | 0.006 ± 0.004 | 0.016 ± 0.004 | -0.05 / -0.14 CKA@L16 helix 0.12 digit 0.18; emb helix 0.65 digit 0.71 |
| digit | 3 | 0.022 ± 0.002 | 0.984 ± 0.007 | 0.001 ± 0.001 | 0.004 ± 0.001 | 0.019 ± 0.003 | 0.22 / 0.61 CKA@L16 helix 0.62 digit 0.99; emb helix 0.65 digit 0.71 |
