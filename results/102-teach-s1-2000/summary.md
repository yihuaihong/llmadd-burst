# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | - |
| main | 3 | 0.044 ± 0.004 | 0.525 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.000 | 0.25 / 0.25 CKA@L16 helix 0.76 digit 0.63 main 0.99 main3 0.64; emb helix 0.30 digit 0.44 |
| main_shuf | 3 | 0.023 ± 0.006 | 0.601 ± 0.025 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.001 | -0.12 / -0.26 CKA@L16 helix 0.12 digit 0.20 main 0.19 main3 0.48; emb helix 0.30 digit 0.44 |
| self | 3 | 0.050 ± 0.007 | 0.448 ± 0.012 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.000 | 0.08 / 0.03 CKA@L16 helix 0.61 digit 0.55 main 0.82 main3 0.63; emb helix 0.30 digit 0.44 |
