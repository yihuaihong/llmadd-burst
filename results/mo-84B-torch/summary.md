# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mixed, split random)

| geom | n seeds | test | train | test_terse | test_sub | test_mul | gpt3_unseen_add | gpt3_unseen_sub | gpt3_unseen_mul | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.081 | 0.096 | 0.016 | 0.072 | 0.132 | 0.018 | 0.054 | 0.108 | - |
| none | 3 | 0.567 ± 0.016 | 0.930 ± 0.012 | 0.091 ± 0.007 | 0.556 ± 0.009 | 0.639 ± 0.009 | 0.130 ± 0.013 | 0.234 ± 0.009 | 0.281 ± 0.005 | 0.29 / 0.36 CKA@L16 helix 0.73 digit 0.56; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.691 ± 0.029 | 0.945 ± 0.009 | 0.174 ± 0.010 | 0.761 ± 0.013 | 0.527 ± 0.004 | 0.215 ± 0.016 | 0.306 ± 0.035 | 0.306 ± 0.013 | 0.73 / 0.73 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.410 ± 0.008 | 0.848 ± 0.008 | 0.094 ± 0.010 | 0.416 ± 0.020 | 0.501 ± 0.022 | 0.092 ± 0.009 | 0.181 ± 0.019 | 0.225 ± 0.029 | -0.08 / -0.21 CKA@L16 helix 0.12 digit 0.17; emb helix 0.65 digit 0.71 |
