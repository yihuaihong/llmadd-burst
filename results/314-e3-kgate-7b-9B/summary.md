# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | - |
| none | 3 | 0.053 ± 0.005 | 0.441 ± 0.006 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.002 | 0.17 / 0.12 CKA@L16 helix 0.69 digit 0.49; emb helix 0.30 digit 0.44 |
| helix | 3 | 0.096 ± 0.005 | 0.557 ± 0.020 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.005 ± 0.001 | 0.62 / 0.58 CKA@L16 helix 1.00 digit 0.64; emb helix 0.30 digit 0.44 |
| sep_ce | 3 | 0.056 ± 0.004 | 0.513 ± 0.044 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.27 / 0.54 CKA@L16 helix 0.63 digit 0.81; emb helix 0.30 digit 0.44 |
| sep_then_helix | 3 | 0.080 ± 0.017 | 0.433 ± 0.029 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.61 / 0.61 CKA@L16 helix 0.99 digit 0.66; emb helix 0.30 digit 0.44 |
