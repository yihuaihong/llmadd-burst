# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mod, split random)

| geom | n seeds | test | train | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|
| base | - | 0.007 | 0.009 | 0.009 | - |
| none | 3 | 0.050 ± 0.008 | 0.145 ± 0.030 | 0.003 ± 0.001 | 0.32 / 0.27 CKA@L16 helix 0.71 digit 0.43; emb helix 0.30 digit 0.44 |
| helix | 3 | 0.074 ± 0.014 | 0.322 ± 0.052 | 0.003 ± 0.001 | 0.68 / 0.64 CKA@L16 helix 1.00 digit 0.63; emb helix 0.30 digit 0.44 |
| helix_shuf | 3 | 0.005 ± 0.001 | 0.031 ± 0.003 | 0.009 ± 0.001 | -0.14 / -0.32 CKA@L16 helix 0.08 digit 0.13; emb helix 0.30 digit 0.44 |
