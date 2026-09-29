# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| none | 2 | 0.569 ± 0.030 | 0.986 ± 0.000 | 0.001 ± 0.002 | 0.007 ± 0.000 | 0.081 ± 0.004 | 0.28 / 0.35 CKA@L16 helix 0.73 digit 0.54; emb helix 0.65 digit 0.71 |
| helix | 2 | 0.687 ± 0.018 | 0.960 ± 0.000 | 0.000 ± 0.000 | 0.007 ± 0.004 | 0.122 ± 0.005 | 0.66 / 0.67 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.71 |
| helix_shuf | 2 | 0.450 ± 0.004 | 0.935 ± 0.004 | 0.000 ± 0.000 | 0.009 ± 0.002 | 0.065 ± 0.006 | -0.05 / -0.14 CKA@L16 helix 0.12 digit 0.18; emb helix 0.65 digit 0.71 |
| digit | 2 | 0.598 ± 0.001 | 0.974 ± 0.007 | 0.001 ± 0.002 | 0.010 ± 0.004 | 0.106 ± 0.010 | 0.22 / 0.61 CKA@L16 helix 0.62 digit 0.99; emb helix 0.65 digit 0.71 |
