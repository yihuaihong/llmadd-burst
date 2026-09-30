# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.011 | 0.013 | 0.000 | 0.000 | 0.001 | - |
| none | 2 | 0.274 ± 0.004 | 0.571 ± 0.047 | 0.004 ± 0.002 | 0.005 ± 0.004 | 0.005 ± 0.001 | 0.28 / 0.34 CKA@L8 helix 0.75 digit 0.59; emb helix 0.66 digit 0.73 |
| helix | 2 | 0.549 ± 0.047 | 0.794 ± 0.040 | 0.001 ± 0.002 | 0.009 ± 0.002 | 0.038 ± 0.001 | 0.62 / 0.63 CKA@L8 helix 0.99 digit 0.66; emb helix 0.66 digit 0.73 |
| helix_shuf | 2 | 0.266 ± 0.011 | 0.575 ± 0.006 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.004 | -0.01 / -0.07 CKA@L8 helix 0.18 digit 0.25; emb helix 0.66 digit 0.73 |
| digit | 2 | 0.498 ± 0.030 | 0.813 ± 0.018 | 0.007 ± 0.004 | 0.001 ± 0.002 | 0.036 ± 0.004 | 0.19 / 0.42 CKA@L8 helix 0.63 digit 0.98; emb helix 0.66 digit 0.73 |
