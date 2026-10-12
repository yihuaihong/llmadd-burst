# manifold fine-tuning (lora, lam 20.0, lr 0.0001, layers [4, 8, 12, 16], task add, split random)

| geom | n seeds | test | train | three_term | three_digit | test_terse | r2 helix / digit (first layer) |
|---|---|---|---|---|---|---|---|
| base | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| helix_shuf | 3 | 0.076 ± 0.004 | 0.168 ± 0.013 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.006 ± 0.001 | -0.13 / -0.30 CKA@L16 helix 0.08 digit 0.13; emb helix 0.30 digit 0.44 |
| digit | 3 | 0.367 ± 0.088 | 0.623 ± 0.076 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.25 / 0.74 CKA@L16 helix 0.62 digit 1.00; emb helix 0.30 digit 0.44 |
