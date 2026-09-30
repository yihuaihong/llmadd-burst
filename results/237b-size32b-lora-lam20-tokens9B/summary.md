# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [8, 16, 24, 32], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.080 ± 0.024 | 0.158 ± 0.050 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.011 ± 0.002 | 0.64 / 0.61 CKA@L32 helix 0.64 digit 0.31; emb helix 0.27 digit 0.43 |
| helix | 3 | 0.066 ± 0.014 | 0.168 ± 0.042 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.011 ± 0.002 | 0.82 / 0.78 CKA@L32 helix 1.00 digit 0.63; emb helix 0.27 digit 0.43 |
| helix_shuf | 3 | 0.056 ± 0.006 | 0.136 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.007 ± 0.001 | -0.13 / -0.30 CKA@L32 helix 0.09 digit 0.13; emb helix 0.27 digit 0.43 |
| digit | 3 | 0.067 ± 0.026 | 0.171 ± 0.047 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.008 ± 0.002 | 0.27 / 0.80 CKA@L32 helix 0.62 digit 1.00; emb helix 0.27 digit 0.43 |
