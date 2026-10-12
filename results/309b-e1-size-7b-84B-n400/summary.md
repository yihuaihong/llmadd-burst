# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.097 | 0.005 | 0.005 | 0.016 | - |
| none | 3 | 0.255 ± 0.006 | 1.000 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.056 ± 0.001 | 0.28 / 0.35 CKA@L16 helix 0.72 digit 0.54; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.296 ± 0.030 | 1.000 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.082 ± 0.007 | 0.72 / 0.72 CKA@L16 helix 1.00 digit 0.64; emb helix 0.65 digit 0.71 |
