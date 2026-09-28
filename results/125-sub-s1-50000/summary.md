# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task sub, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.639 | 0.666 | 0.018 | 0.102 | 0.202 | - |
| none | 3 | 0.964 ± 0.004 | 0.999 ± 0.001 | 0.001 ± 0.001 | 0.177 ± 0.007 | 0.550 ± 0.018 | 0.27 / 0.33 CKA@L16 helix 0.75 digit 0.61; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.975 ± 0.005 | 0.997 ± 0.003 | 0.004 ± 0.001 | 0.144 ± 0.009 | 0.638 ± 0.017 | 0.67 / 0.68 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.897 ± 0.049 | 0.960 ± 0.043 | 0.012 ± 0.005 | 0.109 ± 0.036 | 0.466 ± 0.039 | -0.01 / -0.08 CKA@L16 helix 0.13 digit 0.20; emb helix 0.65 digit 0.72 |
| digit | 3 | 0.965 ± 0.015 | 0.992 ± 0.007 | 0.002 ± 0.004 | 0.141 ± 0.015 | 0.538 ± 0.033 | 0.21 / 0.59 CKA@L16 helix 0.63 digit 0.99; emb helix 0.65 digit 0.72 |
