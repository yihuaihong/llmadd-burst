# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task max, split random)

| geom | n seeds | test | train | test_terse | word0 | word1 | word2 | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|
| base | - | 0.261 | 0.274 | 0.009 | 0.444 | 0.513 | 0.365 | - |
| none | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.848 ± 0.003 | 0.842 ± 0.030 | 0.756 ± 0.006 | 0.874 ± 0.011 | 0.25 / 0.31 CKA@L16 helix 0.72 digit 0.55; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.994 ± 0.005 | 0.998 ± 0.002 | 0.867 ± 0.022 | 0.920 ± 0.025 | 0.831 ± 0.007 | 0.935 ± 0.007 | 0.73 / 0.72 CKA@L16 helix 1.00 digit 0.64; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.986 ± 0.011 | 0.993 ± 0.009 | 0.860 ± 0.023 | 0.692 ± 0.046 | 0.820 ± 0.015 | 0.920 ± 0.011 | -0.10 / -0.24 CKA@L16 helix 0.10 digit 0.15; emb helix 0.60 digit 0.68 |
