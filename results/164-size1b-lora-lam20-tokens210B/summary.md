# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [2, 4, 6, 8], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.039 | 0.053 | 0.005 | 0.002 | 0.001 | - |
| none | 3 | 0.709 ± 0.005 | 0.925 ± 0.007 | 0.002 ± 0.001 | 0.019 ± 0.004 | 0.057 ± 0.010 | 0.23 / 0.27 CKA@L8 helix 0.77 digit 0.63; emb helix 0.70 digit 0.72 |
| helix | 3 | 0.724 ± 0.012 | 0.896 ± 0.009 | 0.002 ± 0.000 | 0.021 ± 0.003 | 0.094 ± 0.010 | 0.56 / 0.56 CKA@L8 helix 0.99 digit 0.65; emb helix 0.70 digit 0.72 |
| helix_shuf | 3 | 0.548 ± 0.045 | 0.843 ± 0.047 | 0.002 ± 0.000 | 0.017 ± 0.001 | 0.078 ± 0.024 | -0.01 / -0.07 CKA@L8 helix 0.15 digit 0.21; emb helix 0.70 digit 0.72 |
| digit | 3 | 0.652 ± 0.034 | 0.879 ± 0.024 | 0.001 ± 0.001 | 0.017 ± 0.004 | 0.115 ± 0.028 | 0.16 / 0.43 CKA@L8 helix 0.62 digit 0.99; emb helix 0.70 digit 0.72 |
