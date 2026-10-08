# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.015 | 0.005 | 0.000 | 0.000 | - |
| none | 3 | 0.175 ± 0.008 | 0.685 ± 0.055 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.005 ± 0.001 | 0.26 / 0.32 CKA@L16 helix 0.72 digit 0.55; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.283 ± 0.023 | 0.760 ± 0.045 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.012 ± 0.004 | 0.64 / 0.63 CKA@L16 helix 1.00 digit 0.65; emb helix 0.60 digit 0.68 |
| digit | 3 | 0.211 ± 0.018 | 0.803 ± 0.017 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.015 ± 0.002 | 0.21 / 0.56 CKA@L16 helix 0.62 digit 0.99; emb helix 0.60 digit 0.68 |
| helix_nonsep | 3 | 0.121 ± 0.015 | 0.492 ± 0.021 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.007 ± 0.002 | 0.18 / 0.09 CKA@L16 helix 0.54 digit 0.34; emb helix 0.60 digit 0.68 |
| helix_coarse | 3 | 0.141 ± 0.021 | 0.458 ± 0.006 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.008 ± 0.005 | 0.65 / 0.65 CKA@L16 helix 0.70 digit 0.36; emb helix 0.60 digit 0.68 |
| sep_ce | 3 | 0.216 ± 0.033 | 0.766 ± 0.065 | 0.000 ± 0.000 | 0.002 ± 0.002 | 0.013 ± 0.003 | 0.33 / 0.53 CKA@L16 helix 0.69 digit 0.81; emb helix 0.60 digit 0.68 |
