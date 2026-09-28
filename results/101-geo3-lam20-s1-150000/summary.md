# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16])

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.878 | 0.882 | 0.045 | 0.477 | 0.585 | - |
| helix3 | 3 | 0.975 ± 0.013 | 0.986 ± 0.011 | 0.072 ± 0.008 | 0.222 ± 0.082 | 0.897 ± 0.023 | 0.57 / 0.59 CKA@L16 helix 0.96 digit 0.71; emb helix 0.70 digit 0.71 |
| helix3_shuf | 3 | 0.989 ± 0.003 | 0.996 ± 0.002 | 0.078 ± 0.023 | 0.238 ± 0.014 | 0.911 ± 0.012 | 0.09 / 0.07 CKA@L16 helix 0.30 digit 0.34; emb helix 0.70 digit 0.71 |
| digit3 | 3 | 0.991 ± 0.004 | 0.995 ± 0.003 | 0.077 ± 0.015 | 0.414 ± 0.059 | 0.949 ± 0.005 | 0.26 / 0.42 CKA@L16 helix 0.71 digit 0.93; emb helix 0.70 digit 0.71 |
| digit3_shuf | 3 | 0.974 ± 0.005 | 0.985 ± 0.007 | 0.073 ± 0.012 | 0.096 ± 0.025 | 0.917 ± 0.021 | 0.07 / 0.04 CKA@L16 helix 0.48 digit 0.58; emb helix 0.70 digit 0.71 |
