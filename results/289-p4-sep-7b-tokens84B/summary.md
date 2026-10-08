# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.084 | 0.092 | 0.005 | 0.005 | 0.016 | - |
| none | 3 | 0.566 ± 0.025 | 0.979 ± 0.008 | 0.000 ± 0.000 | 0.010 ± 0.002 | 0.078 ± 0.002 | 0.28 / 0.35 CKA@L16 helix 0.73 digit 0.54; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.664 ± 0.048 | 0.948 ± 0.022 | 0.001 ± 0.001 | 0.007 ± 0.002 | 0.118 ± 0.006 | 0.66 / 0.67 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.71 |
| digit | 3 | 0.593 ± 0.013 | 0.968 ± 0.006 | 0.000 ± 0.000 | 0.008 ± 0.001 | 0.106 ± 0.009 | 0.22 / 0.60 CKA@L16 helix 0.62 digit 0.99; emb helix 0.65 digit 0.71 |
| helix_nonsep | 3 | 0.489 ± 0.031 | 0.918 ± 0.033 | 0.000 ± 0.000 | 0.012 ± 0.003 | 0.067 ± 0.007 | 0.22 / 0.15 CKA@L16 helix 0.56 digit 0.37; emb helix 0.65 digit 0.71 |
| helix_coarse | 3 | 0.529 ± 0.000 | 0.940 ± 0.005 | 0.000 ± 0.000 | 0.015 ± 0.007 | 0.087 ± 0.001 | 0.65 / 0.66 CKA@L16 helix 0.71 digit 0.37; emb helix 0.65 digit 0.71 |
| sep_ce | 3 | 0.607 ± 0.013 | 0.971 ± 0.007 | 0.000 ± 0.000 | 0.010 ± 0.002 | 0.109 ± 0.012 | 0.33 / 0.56 CKA@L16 helix 0.73 digit 0.84; emb helix 0.65 digit 0.71 |
