# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [8, 16, 24, 32], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.030 | 0.034 | 0.007 | 0.000 | 0.001 | - |
| none | 3 | 0.278 ± 0.012 | 0.719 ± 0.038 | 0.001 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.18 / 0.17 CKA@L32 helix 0.74 digit 0.58; emb helix 0.27 digit 0.44 |
| helix | 3 | 0.321 ± 0.018 | 0.680 ± 0.015 | 0.002 ± 0.001 | 0.005 ± 0.002 | 0.013 ± 0.001 | 0.59 / 0.56 CKA@L32 helix 0.99 digit 0.65; emb helix 0.27 digit 0.44 |
| helix_shuf | 3 | 0.186 ± 0.007 | 0.533 ± 0.029 | 0.003 ± 0.001 | 0.002 ± 0.000 | 0.009 ± 0.003 | -0.10 / -0.26 CKA@L32 helix 0.11 digit 0.17; emb helix 0.27 digit 0.44 |
| digit | 3 | 0.261 ± 0.027 | 0.687 ± 0.045 | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.014 ± 0.003 | 0.18 / 0.54 CKA@L32 helix 0.62 digit 0.99; emb helix 0.27 digit 0.44 |
