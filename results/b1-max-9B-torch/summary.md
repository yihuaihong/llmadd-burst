# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task max, split random)

| geom | n seeds | test | train | test_terse | word0 | word1 | word2 | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|---|
| base | - | 0.075 | 0.089 | 0.003 | 0.010 | 0.096 | 0.016 | - |
| none | 3 | 0.979 ± 0.001 | 0.998 ± 0.003 | 0.073 ± 0.014 | 0.560 ± 0.012 | 0.735 ± 0.020 | 0.688 ± 0.015 | 0.10 / 0.04 CKA@L16 helix 0.65 digit 0.54; emb helix 0.30 digit 0.44 |
| helix | 3 | 0.964 ± 0.005 | 0.986 ± 0.008 | 0.240 ± 0.006 | 0.723 ± 0.025 | 0.880 ± 0.017 | 0.816 ± 0.029 | 0.64 / 0.59 CKA@L16 helix 1.00 digit 0.63; emb helix 0.30 digit 0.44 |
| helix_shuf | 3 | 0.938 ± 0.010 | 0.988 ± 0.007 | 0.133 ± 0.028 | 0.560 ± 0.029 | 0.718 ± 0.046 | 0.696 ± 0.034 | -0.12 / -0.29 CKA@L16 helix 0.09 digit 0.14; emb helix 0.30 digit 0.44 |
