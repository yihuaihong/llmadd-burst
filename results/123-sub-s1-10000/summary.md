# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task sub, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.014 | 0.021 | 0.047 | 0.005 | 0.009 | - |
| none | 3 | 0.273 ± 0.034 | 0.626 ± 0.097 | 0.021 ± 0.004 | 0.008 ± 0.001 | 0.015 ± 0.001 | 0.28 / 0.33 CKA@L16 helix 0.73 digit 0.54; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.397 ± 0.074 | 0.723 ± 0.095 | 0.010 ± 0.007 | 0.012 ± 0.004 | 0.016 ± 0.001 | 0.59 / 0.59 CKA@L16 helix 0.99 digit 0.65; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.161 ± 0.015 | 0.434 ± 0.056 | 0.018 ± 0.010 | 0.007 ± 0.001 | 0.016 ± 0.000 | -0.03 / -0.12 CKA@L16 helix 0.11 digit 0.17; emb helix 0.60 digit 0.68 |
| digit | 3 | 0.303 ± 0.013 | 0.725 ± 0.042 | 0.006 ± 0.005 | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.21 / 0.54 CKA@L16 helix 0.62 digit 0.99; emb helix 0.60 digit 0.68 |
