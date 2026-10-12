# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mod, split random)

| geom | n seeds | test | train | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|
| base | - | 0.012 | 0.013 | 0.007 | - |
| none | 3 | 0.950 ± 0.015 | 0.995 ± 0.005 | 0.156 ± 0.011 | 0.27 / 0.34 CKA@L16 helix 0.75 digit 0.62; emb helix 0.65 digit 0.72 |
| helix | 3 | 0.883 ± 0.040 | 0.977 ± 0.021 | 0.198 ± 0.010 | 0.64 / 0.66 CKA@L16 helix 0.99 digit 0.66; emb helix 0.65 digit 0.72 |
| helix_shuf | 3 | 0.840 ± 0.032 | 0.964 ± 0.014 | 0.182 ± 0.032 | -0.02 / -0.09 CKA@L16 helix 0.14 digit 0.21; emb helix 0.65 digit 0.72 |
