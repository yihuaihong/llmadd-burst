# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mixed, split random)

| geom | n seeds | test | train | test_terse | test_sub | test_mul | gpt3_unseen_add | gpt3_unseen_sub | gpt3_unseen_mul | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | - | 0.007 | 0.018 | 0.000 | 0.016 | 0.028 | 0.001 | 0.004 | 0.000 | - |
| none | 3 | 0.189 ± 0.010 | 0.713 ± 0.018 | 0.015 ± 0.005 | 0.301 ± 0.019 | 0.392 ± 0.007 | 0.005 ± 0.001 | 0.015 ± 0.003 | 0.075 ± 0.008 | 0.29 / 0.34 CKA@L16 helix 0.74 digit 0.54; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.318 ± 0.014 | 0.796 ± 0.017 | 0.018 ± 0.003 | 0.490 ± 0.002 | 0.274 ± 0.029 | 0.005 ± 0.000 | 0.018 ± 0.005 | 0.064 ± 0.013 | 0.70 / 0.70 CKA@L16 helix 1.00 digit 0.64; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.119 ± 0.006 | 0.540 ± 0.009 | 0.022 ± 0.003 | 0.190 ± 0.016 | 0.245 ± 0.013 | 0.003 ± 0.003 | 0.020 ± 0.014 | 0.056 ± 0.005 | -0.08 / -0.21 CKA@L16 helix 0.10 digit 0.15; emb helix 0.60 digit 0.68 |
