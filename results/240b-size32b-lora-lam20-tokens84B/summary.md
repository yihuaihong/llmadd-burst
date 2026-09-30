# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [8, 16, 24, 32], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.158 | 0.162 | 0.010 | 0.000 | 0.032 | - |
| none | 3 | 0.684 ± 0.006 | 0.980 ± 0.003 | 0.001 ± 0.001 | 0.012 ± 0.003 | 0.108 ± 0.008 | 0.12 / 0.09 CKA@L32 helix 0.74 digit 0.60; emb helix 0.28 digit 0.44 |
| helix | 3 | 0.812 ± 0.032 | 0.966 ± 0.017 | 0.003 ± 0.001 | 0.012 ± 0.005 | 0.179 ± 0.013 | 0.55 / 0.52 CKA@L32 helix 0.99 digit 0.65; emb helix 0.28 digit 0.44 |
| helix_shuf | 3 | 0.525 ± 0.035 | 0.896 ± 0.032 | 0.012 ± 0.009 | 0.008 ± 0.001 | 0.116 ± 0.004 | -0.08 / -0.21 CKA@L32 helix 0.13 digit 0.20; emb helix 0.28 digit 0.44 |
| digit | 3 | 0.785 ± 0.041 | 0.981 ± 0.015 | 0.002 ± 0.002 | 0.020 ± 0.002 | 0.163 ± 0.019 | 0.18 / 0.51 CKA@L32 helix 0.62 digit 0.99; emb helix 0.28 digit 0.44 |
