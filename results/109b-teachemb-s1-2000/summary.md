# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | - |
| main | 3 | 0.053 ± 0.008 | 0.462 ± 0.035 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.33 / 0.33 CKA@L16 helix 0.71 digit 0.48 main 0.88 main3 0.63; emb helix 0.67 digit 0.67 |
| main_shuf | 3 | 0.044 ± 0.004 | 0.444 ± 0.018 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.18 / 0.12 CKA@L16 helix 0.68 digit 0.46 main 0.85 main3 0.63; emb helix 0.20 digit 0.32 |
