# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split holdout_operand, geo_numbers seen)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.105 | 0.078 | 0.005 | 0.005 | 0.025 | - |
| helix | 3 | 0.571 ± 0.010 | 0.940 ± 0.018 | 0.000 ± 0.000 | 0.008 ± 0.001 | 0.140 ± 0.013 | 0.64 / 0.65 CKA@L16 helix 0.98 digit 0.64; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.394 ± 0.011 | 0.930 ± 0.004 | 0.002 ± 0.000 | 0.010 ± 0.005 | 0.095 ± 0.013 | -0.02 / -0.05 CKA@L16 helix 0.14 digit 0.22; emb helix 0.65 digit 0.71 |
