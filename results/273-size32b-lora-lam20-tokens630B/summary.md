# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [8, 16, 24, 32], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.979 | 0.984 | 0.218 | 0.817 | 0.752 | - |
| none | 3 | 1.000 ± 0.001 | 1.000 ± 0.000 | 0.240 ± 0.020 | 0.904 ± 0.019 | 0.873 ± 0.022 | 0.09 / 0.07 CKA@L32 helix 0.68 digit 0.61; emb helix 0.31 digit 0.47 |
| helix | 3 | 0.995 ± 0.002 | 0.999 ± 0.001 | 0.264 ± 0.019 | 0.807 ± 0.035 | 0.874 ± 0.031 | 0.67 / 0.65 CKA@L32 helix 1.00 digit 0.65; emb helix 0.31 digit 0.47 |
| helix_shuf | 3 | 0.971 ± 0.036 | 0.978 ± 0.028 | 0.183 ± 0.012 | 0.807 ± 0.024 | 0.841 ± 0.018 | -0.08 / -0.21 CKA@L32 helix 0.12 digit 0.18; emb helix 0.31 digit 0.47 |
| digit | 3 | 0.994 ± 0.004 | 1.000 ± 0.001 | 0.217 ± 0.020 | 0.830 ± 0.016 | 0.884 ± 0.023 | 0.23 / 0.69 CKA@L32 helix 0.62 digit 1.00; emb helix 0.31 digit 0.47 |
