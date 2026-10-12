# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task mod, split random)

| geom | n seeds | test | train | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|
| base | - | 0.012 | 0.009 | 0.007 | - |
| none | 3 | 0.245 ± 0.012 | 0.737 ± 0.029 | 0.006 ± 0.002 | 0.27 / 0.32 CKA@L16 helix 0.73 digit 0.55; emb helix 0.60 digit 0.68 |
| helix | 3 | 0.450 ± 0.016 | 0.869 ± 0.026 | 0.009 ± 0.001 | 0.63 / 0.63 CKA@L16 helix 0.99 digit 0.65; emb helix 0.60 digit 0.68 |
| helix_shuf | 3 | 0.133 ± 0.056 | 0.455 ± 0.214 | 0.007 ± 0.002 | -0.07 / -0.19 CKA@L16 helix 0.10 digit 0.16; emb helix 0.60 digit 0.68 |
