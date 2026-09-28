# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split holdout_operand)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.105 | 0.078 | 0.005 | 0.005 | 0.025 | - |
| none | 3 | 0.484 ± 0.037 | 0.981 ± 0.011 | 0.001 ± 0.001 | 0.012 ± 0.004 | 0.094 ± 0.006 | 0.28 / 0.35 CKA@L16 helix 0.73 digit 0.54; emb helix 0.65 digit 0.71 |
| helix | 3 | 0.601 ± 0.048 | 0.957 ± 0.013 | 0.000 ± 0.000 | 0.007 ± 0.004 | 0.135 ± 0.010 | 0.66 / 0.67 CKA@L16 helix 1.00 digit 0.65; emb helix 0.65 digit 0.71 |
| helix_shuf | 3 | 0.333 ± 0.037 | 0.925 ± 0.027 | 0.002 ± 0.003 | 0.012 ± 0.009 | 0.081 ± 0.004 | -0.05 / -0.14 CKA@L16 helix 0.12 digit 0.18; emb helix 0.65 digit 0.71 |
| digit | 3 | 0.505 ± 0.039 | 0.956 ± 0.012 | 0.002 ± 0.003 | 0.004 ± 0.003 | 0.114 ± 0.003 | 0.22 / 0.60 CKA@L16 helix 0.62 digit 0.99; emb helix 0.65 digit 0.71 |
