# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task max, split random)

| geom | n seeds | test | train | test_terse | word0 | word1 | word2 | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|
| base | - | 0.040 | 0.038 | 0.092 | 0.325 | 0.576 | 0.535 | - |
| none | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.790 ± 0.028 | 0.937 ± 0.014 | 0.876 ± 0.023 | 0.915 ± 0.024 | 0.27 / 0.33 CKA@L16 helix 0.73 digit 0.56; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.989 ± 0.012 | 0.990 ± 0.013 | 0.933 ± 0.009 | 0.966 ± 0.008 | 0.945 ± 0.010 | 0.939 ± 0.007 | 0.72 / 0.72 CKA@L16 helix 1.00 digit 0.64; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.988 ± 0.007 | 0.994 ± 0.005 | 0.789 ± 0.040 | 0.962 ± 0.014 | 0.929 ± 0.022 | 0.934 ± 0.010 | -0.07 / -0.17 CKA@L16 helix 0.09 digit 0.15; emb helix 0.65 digit 0.72 |
