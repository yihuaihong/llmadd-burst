# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [8, 16, 24, 32], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.857 | 0.847 | 0.062 | 0.138 | 0.295 | - |
| none | 3 | 0.995 ± 0.002 | 1.000 ± 0.000 | 0.027 ± 0.008 | 0.251 ± 0.009 | 0.648 ± 0.009 | 0.07 / 0.01 CKA@L32 helix 0.74 digit 0.59; emb helix 0.29 digit 0.45 |
| helix | 3 | 0.977 ± 0.010 | 0.997 ± 0.003 | 0.043 ± 0.008 | 0.204 ± 0.010 | 0.757 ± 0.012 | 0.56 / 0.52 CKA@L32 helix 1.00 digit 0.65; emb helix 0.29 digit 0.45 |
| helix_shuf | 3 | 0.917 ± 0.006 | 0.968 ± 0.008 | 0.047 ± 0.015 | 0.207 ± 0.016 | 0.571 ± 0.030 | -0.09 / -0.23 CKA@L32 helix 0.12 digit 0.18; emb helix 0.29 digit 0.45 |
| digit | 3 | 0.987 ± 0.005 | 1.000 ± 0.000 | 0.057 ± 0.013 | 0.227 ± 0.002 | 0.792 ± 0.022 | 0.18 / 0.54 CKA@L32 helix 0.62 digit 1.00; emb helix 0.29 digit 0.45 |
