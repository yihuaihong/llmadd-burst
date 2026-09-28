# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split carry)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.011 | 0.013 | 0.005 | 0.000 | 0.002 | - |
| none | 3 | 0.001 ± 0.001 | 0.931 ± 0.022 | 0.001 ± 0.001 | 0.004 ± 0.001 | 0.001 ± 0.001 | 0.26 / 0.32 CKA@L16 helix 0.73 digit 0.57; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.000 ± 0.001 | 0.890 ± 0.015 | 0.000 ± 0.000 | 0.003 ± 0.001 | 0.000 ± 0.000 | 0.64 / 0.64 CKA@L16 helix 1.00 digit 0.65; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.002 ± 0.002 | 0.808 ± 0.014 | 0.000 ± 0.000 | 0.005 ± 0.002 | 0.001 ± 0.000 | -0.04 / -0.13 CKA@L16 helix 0.11 digit 0.17; emb helix 0.60 digit 0.68 |
| digit | 3 | 0.000 ± 0.000 | 0.951 ± 0.016 | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.000 ± 0.000 | 0.21 / 0.58 CKA@L16 helix 0.62 digit 0.99; emb helix 0.60 digit 0.68 |
