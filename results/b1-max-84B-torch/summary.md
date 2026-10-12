# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task max, split random)

| geom | n seeds | test | train | test_terse | word0 | word1 | word2 | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|
| base | - | 0.290 | 0.301 | 0.335 | 0.381 | 0.594 | 0.489 | - |
| none | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.910 ± 0.020 | 0.952 ± 0.010 | 0.871 ± 0.011 | 0.790 ± 0.020 | 0.28 / 0.35 CKA@L16 helix 0.72 digit 0.52; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.997 ± 0.001 | 1.000 ± 0.000 | 0.941 ± 0.003 | 0.952 ± 0.007 | 0.917 ± 0.006 | 0.834 ± 0.019 | 0.74 / 0.75 CKA@L16 helix 1.00 digit 0.64; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.980 ± 0.003 | 0.987 ± 0.009 | 0.906 ± 0.042 | 0.965 ± 0.011 | 0.893 ± 0.024 | 0.855 ± 0.025 | -0.09 / -0.21 CKA@L16 helix 0.09 digit 0.14; emb helix 0.65 digit 0.71 |
