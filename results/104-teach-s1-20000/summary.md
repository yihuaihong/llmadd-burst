# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| main | 3 | 0.594 ± 0.008 | 0.977 ± 0.007 | 0.000 ± 0.000 | 0.010 ± 0.004 | 0.085 ± 0.003 | 0.32 / 0.39 CKA@L16 helix 0.76 digit 0.63 main 0.99 main3 0.85; emb helix 0.65 digit 0.71 |
| main_shuf | 3 | 0.460 ± 0.015 | 0.948 ± 0.002 | 0.000 ± 0.000 | 0.013 ± 0.001 | 0.073 ± 0.011 | -0.02 / -0.06 CKA@L16 helix 0.16 digit 0.25 main 0.23 main3 0.69; emb helix 0.65 digit 0.71 |
| self | 3 | 0.557 ± 0.017 | 0.978 ± 0.012 | 0.000 ± 0.000 | 0.014 ± 0.003 | 0.076 ± 0.002 | 0.28 / 0.35 CKA@L16 helix 0.72 digit 0.55 main 0.92 main3 0.82; emb helix 0.65 digit 0.71 |
