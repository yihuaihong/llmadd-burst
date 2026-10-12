# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mod, split random)

| geom | n seeds | test | train | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|
| base | - | 0.009 | 0.009 | 0.005 | - |
| none | 3 | 0.670 ± 0.013 | 0.980 ± 0.004 | 0.002 ± 0.002 | 0.28 / 0.35 CKA@L16 helix 0.74 digit 0.55; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.652 ± 0.116 | 0.902 ± 0.094 | 0.005 ± 0.001 | 0.65 / 0.66 CKA@L16 helix 0.99 digit 0.66; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.523 ± 0.017 | 0.929 ± 0.007 | 0.004 ± 0.001 | -0.03 / -0.11 CKA@L16 helix 0.13 digit 0.19; emb helix 0.65 digit 0.71 |
